import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union, cast

logger = logging.getLogger(__name__)

from nlqs.database.postgres import PostgresConnectionConfig, PostgresDriver
from nlqs.database.sqlite import SQLiteConnectionConfig, SQLiteDriver
from nlqs.parameters import DEFAULT_DB_NAME, DEFAULT_TABLE_NAME
from nlqs.query_construction import (
    construct_categorical_search_query_fragments,
    construct_descriptive_search_query_fragments,
    construct_quantitaive_search_query_fragments,
    construct_identifier_search_query_fragments,
)
from nlqs.summarization import summarize
from nlqs.vectordb_driver import ChromaDBConfig, VectorDBDriver
from nlqs.search_field import SearchField

# Add Neon imports
from nlqs.neondb_driver import NeonDBConfig, NeonVectorDBDriver
from utils.llm import get_default_llm, get_default_embedding_function


@dataclass
class NLQSResult:
    records: List[Dict[str, Any]]
    uris: List[str]
    is_input_irrelevant: bool = False
    # Whether the records are an exact match (intersection of all constraints) or a
    # "related" fallback (union of constraints). True when there are no records.
    is_exact_match: bool = True


class NLQS:
    def __init__(
        self,
        connection_config: Union[SQLiteConnectionConfig, PostgresConnectionConfig],
        chroma_config: Union[ChromaDBConfig, NeonDBConfig],
        use_azure_llm: bool = True,
        use_local_embeddings: bool = True,
    ) -> None:
        logger.info("Initializing NLQS...")

        # Initialize database connection
        self.connection_config = connection_config
        # Store vector backend configuration (can be Chroma or Neon)
        self.chroma_config = chroma_config
        if isinstance(connection_config, SQLiteConnectionConfig):
            logger.info("Using SQLite database")
            self.connection_driver = SQLiteDriver(connection_config)
        elif isinstance(connection_config, PostgresConnectionConfig):
            logger.info("Using PostgreSQL database")
            self.connection_driver = PostgresDriver(connection_config)
        else:
            logger.error("Invalid connection configuration")
            raise ValueError("Invalid connection configuration")

        # Initialize the connection to the database
        logger.debug("Connecting to database...")
        self.connection_driver.connect()
        logger.info("Database connection established")

        # Create the llm object
        logger.debug("Initializing LLM...")
        self.llm = get_default_llm(use_azure=use_azure_llm)
        logger.info("LLM initialized")

        # Initialize the Embedding model (local BGE by default)
        logger.debug("Initializing embedding model...")
        embedding_model = get_default_embedding_function(
            use_local=use_local_embeddings, use_azure=not use_local_embeddings
        )
        embedding_function = embedding_model.embed_query
        logger.info("Embedding model initialized")

        # Initialize the vector backend: Chroma (default) or Neon if NeonDBConfig passed
        if isinstance(self.chroma_config, ChromaDBConfig):
            self.vectordb_driver = VectorDBDriver(self.chroma_config, embedding_function=embedding_function)
        elif isinstance(self.chroma_config, NeonDBConfig):
            self.vectordb_driver = NeonVectorDBDriver(self.chroma_config, embedding_function=embedding_function)
        else:
            raise ValueError("Invalid vector backend configuration")

        logger.info("Vector database initialized")

        self.table_name = connection_config.dataset_table_name
        self.uri_column = connection_config.uri_column
        self.output_columns = connection_config.output_columns

        # Identifiers used to tag/filter vectors for this dataset. Decoupled from the
        # SQL connection identifiers; default to the NLQS defaults when not configured.
        self.vector_db_name = getattr(self.chroma_config, "vector_db_name", DEFAULT_DB_NAME)
        self.vector_table_name = getattr(self.chroma_config, "vector_table_name", DEFAULT_TABLE_NAME)

        # Test if all infrastructure is available
        logger.debug("Checking vector collections...")
        if (
            hasattr(self.vectordb_driver, "check_nlqs_collections_exists")
            and self.vectordb_driver.check_nlqs_collections_exists() is False
        ):
            logger.error("Vector collections do not exist")
            raise ValueError("Vector collections do not exist. Please create them.")
        logger.info("Vector collections verified")

    def execute_nlqs_query_workflow(self, user_input: str, chat_history: List[Tuple[str, str]]) -> NLQSResult:
        logger.info(f"Executing NLQS query workflow for input: {user_input}")

        # Step 0 - Create the pre-requisite objects
        driver = self.connection_driver

        # Retrieve descriptions and types from db
        logger.debug("Retrieving column descriptions from vector database...")
        column_descriptions_dict = self.vectordb_driver.retrieve_descriptions_and_types_from_db()

        logger.debug(f"column_descriptions_dict: {column_descriptions_dict}")

        if column_descriptions_dict is None:
            logger.error("No data found in the database")
            raise ValueError("No data found in the database. Generate Column descriptions.")
        logger.debug("Column descriptions retrieved successfully")

        # Get the primary key for the table
        primary_key = driver.get_primary_key(self.table_name)
        logger.debug(f"Using primary key: {primary_key}")

        # Get Vector backend readiness (Chroma exposes dataset_collection; Neon does not)
        logger.debug("Validating vector backend...")
        if hasattr(self.vectordb_driver, "dataset_collection"):
            chroma_data_collection = getattr(self.vectordb_driver, "dataset_collection", None)
            if chroma_data_collection is None:
                logger.error("Chroma Collection not found")
                raise ValueError("Chroma Collection not found in vectordb. Please create a collection.")
        logger.debug("Vector backend ready")

        # Step 5 - check if the user input is empty
        if not user_input.strip():
            logger.info("Empty user input received")
            return NLQSResult(records=[], uris=[])

        # Step 6 - Remove curly braces from input
        logger.debug("Processing user input...")
        user_input = re.sub(r"{|}", "", user_input)

        # Step 7 - Generate summary
        logger.debug("Generating input summary...")

        def _summarize() -> Any:
            return summarize(
                user_input=user_input,
                chat_history=chat_history,
                column_descriptions_dictionary=column_descriptions_dict["column_descriptions"],
                numerical_columns=column_descriptions_dict["numerical_columns"],
                categorical_columns=column_descriptions_dict["categorical_columns"],
                descriptive_columns=column_descriptions_dict["descriptive_columns"],
                llm=self.llm,
                vectordb=cast(Any, self.vectordb_driver),
                identifier_columns=column_descriptions_dict.get("identifier_columns", []),
                db_name=self.vector_db_name,
                table_name=self.vector_table_name,
            )

        try:
            summarized_input = _summarize()
            logger.debug(f"Generated summary: {summarized_input}")
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}", exc_info=True)
            raise

        intent = summarized_input.user_intent

        logger.info("--------------------------")
        logger.info(f"user input: {user_input}")
        logger.info(f"Summarized input: {summarized_input}")

        # Handle non-data intents before anything else so that greetings and
        # malicious inputs are short-circuited instead of being retried.
        if intent == "sql_injection":
            # Kill the workflow if the user input is a SQL injection
            return NLQSResult(records=[], uris=[], is_input_irrelevant=True)
        elif intent == "phatic_communication":
            # Kill the workflow if the user input is phatic communication
            return NLQSResult(records=[], uris=[], is_input_irrelevant=True)

        # Retry summarization a bounded number of times if no summary was produced.
        # An empty summary after retries is treated as an irrelevant input rather
        # than a hard failure.
        count = 0
        while not summarized_input.summary and count < 5:
            summarized_input = _summarize()
            count += 1

        if not summarized_input.summary:
            logger.info("Unable to summarize the input; treating it as irrelevant.")
            return NLQSResult(records=[], uris=[], is_input_irrelevant=True)

        # This is the standard workflow for the NLQS

        # TODO: We should reenable this        # # Check if the user requested columns exist
        # for column in summarized_input.user_requested_columns:
        #     if column not in column_descriptions:
        #         raise ValueError(f"Column {column} not found in the database.")

        logger.debug("checking for user requested columns...")
        if len(summarized_input.user_requested_columns) > 0:
            numerical_data = summarized_input.numerical_data
            categorical_data = summarized_input.categorical_data
            descriptive_data = summarized_input.descriptive_data
            identifier_data = summarized_input.identifier_data

            # Pass the LLM instance to the quantitative query construction functions
            logger.debug("Constructing query fragments...")
            identifier_query_fragments = construct_identifier_search_query_fragments(identifier_data)
            quantitative_query_fragments = construct_quantitaive_search_query_fragments(numerical_data, self.llm)
            categorical_query_fragments = construct_categorical_search_query_fragments(categorical_data)
            descriptive_query_fragments = construct_descriptive_search_query_fragments(
                descriptive_data,
                cast(Any, self.vectordb_driver),
                db_name=self.vector_db_name,
                table_name=self.vector_table_name,
            )

            logger.debug(
                f"Query fragments constructed - "
                f"Identifier: {len(identifier_query_fragments)}, "
                f"Quantitative: {len(quantitative_query_fragments)}, "
                f"Categorical: {len(categorical_query_fragments)}, "
                f"Descriptive: {len(descriptive_query_fragments)}"
            )

            # Construct a search field that runs the per-field-type queries, intersects
            # the resulting primary keys (exact match) and falls back to a union
            # (related match) when the intersection is empty.
            search_field_object = SearchField.construct_search_field(
                descriptive_query_fragments=[
                    fragment for fragments in descriptive_query_fragments.values() for fragment in fragments
                ],
                categorical_query_fragments=categorical_query_fragments,
                identifier_query_fragments=identifier_query_fragments,
                quantitative_query_fragments=quantitative_query_fragments,
                database_driver=self.connection_driver,
                database_name=self.vector_db_name,
                table_name=self.table_name,  # Use actual table name from config
                primary_key=primary_key,
            )

            # Aggregated primary keys and whether they were an exact match
            unique_primary_keys = list(dict.fromkeys(search_field_object.get_primary_keys()))
            is_exact_match = search_field_object.is_exact_match
            logger.debug(f"Search produced {len(unique_primary_keys)} primary keys (exact={is_exact_match})")

            if not unique_primary_keys:
                logger.info("No matching records found")
                result = NLQSResult(records=[], uris=[])
            else:
                # Convert primary keys to string for SQL query
                primary_keys_string = ",".join(str(pk) for pk in unique_primary_keys)

                # Get the columns in the order they appear in the database
                columns_database = driver.get_database_columns(self.table_name)

                # Variables for specific columns
                uri_column = self.uri_column
                output_columns = self.output_columns

                # If output_columns is specified, modify the query to select only those columns
                if output_columns:
                    # Ensure primary key is included for processing
                    columns_to_select = output_columns.copy()
                    if primary_key not in columns_to_select:
                        columns_to_select.append(primary_key)
                    if uri_column and uri_column not in columns_to_select:
                        columns_to_select.append(uri_column)

                    final_query = f"SELECT {','.join(col for col in columns_to_select)} FROM {self.table_name} WHERE {primary_key} IN ({primary_keys_string})"
                    data_retrieved = driver.execute_query(final_query)
                    columns_to_use = columns_to_select
                else:
                    # Execute the query to retrieve the data with all columns
                    final_query = f"SELECT * FROM {self.table_name} WHERE {primary_key} IN ({primary_keys_string})"
                    data_retrieved = driver.execute_query(final_query)
                    columns_to_use = columns_database

                # Initialize lists to hold records and URIs
                records = []
                uris = []

                if not data_retrieved:
                    result = NLQSResult(records=[], uris=[], is_exact_match=is_exact_match)
                else:
                    # Process the retrieved data
                    for row in data_retrieved:
                        record = dict(zip(columns_to_use, row))

                        # Extract URI if specified
                        if uri_column and uri_column in record:
                            uris.append(str(record[uri_column]))
                            if uri_column != primary_key:  # Don't delete primary key
                                del record[uri_column]

                        # Remove primary key from record if it's not in output_columns
                        if output_columns and primary_key in record and primary_key not in output_columns:
                            del record[primary_key]

                        records.append(record)

                    # Create the result object
                    result = NLQSResult(records=records, uris=uris, is_exact_match=is_exact_match)

                logger.info(f"Query executed: {final_query}")
                logger.info(f"Found {len(records)} records (exact_match={is_exact_match})")
        else:
            logger.info("No user requested columns found")
            result = NLQSResult(records=[], uris=[])

        return result
