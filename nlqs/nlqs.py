import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from langchain_openai import AzureOpenAIEmbeddings, ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from nlqs.database.postgres import PostgresConnectionConfig, PostgresDriver
from nlqs.database.sqlite import SQLiteConnectionConfig, SQLiteDriver
from nlqs.parameters import DEFAULT_DB_NAME, DEFAULT_TABLE_NAME
from nlqs.query_construction import (
    construct_categorical_search_query_fragments,
    construct_descriptive_search_query_fragments,
    construct_final_search_query,
    construct_quantitaive_search_query_fragments,
)
from nlqs.summarization import summarize
from nlqs.vectordb_driver import ChromaDBConfig, VectorDBDriver
from nlqs.search_field import SearchField
from utils.llm import get_default_llm, get_default_embedding_function

# Create a logger object
logger = logging.getLogger(__name__)

# Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR)
logger.setLevel(logging.INFO)

# Create a stream handler to output logs to the console
stream_handler = logging.StreamHandler()

# Create a formatter to format the log messages
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Add the formatter to the stream handler
stream_handler.setFormatter(formatter)

# Add the stream handler to the logger
logger.addHandler(stream_handler)


@dataclass
class NLQSResult:
    records: List[Dict[str, Any]]
    uris: List[str]
    is_input_irrelevant: bool = False


class NLQS:
    def __init__(
        self, connection_config: Union[SQLiteConnectionConfig, PostgresConnectionConfig], chroma_config: ChromaDBConfig
    ) -> None:
        logger.info("Initializing NLQS...")

        self.connection_driver = self._get_connection_driver(connection_config)

        # Initialize the connection to the database
        logger.debug("Connecting to database...")
        self.connection_driver.connect()
        logger.info("Database connection established")

        # Create the llm object
        self.llm = self._iniitalize_llm()

        # Intialize vector DB driver
        self.vectordb_driver = self._initialize_vectordb_driver(chroma_config )

        self.table_name = connection_config.dataset_table_name
        self.uri_column = connection_config.uri_column
        self.output_columns = connection_config.output_columns

        # Test infrastructure
        self._test_infrastructure()
        
    def execute_nlqs_query_workflow(self, user_input: str, chat_history: List[Tuple[str, str]]) -> NLQSResult:
        """This function is where the whole interaction happens.
        It takes the user input and chat history as input and returns the response if the user's intent is either phatic_communication, profanity or sql_injection.
        Else it returns the query result or search similarity result.

        Args:
            user_input (str): The user's input.
            chat_history (list[(str, str)]): The chat history.

        Returns:
            result (NLQSResult): The result
        """

        # Overview
        # Step 1 - retrieve descriptions and types from db. check if its empty. if not return the data.
        # Step 2 - else if the retrived data was empty then generate new columns descriptions.
        # Step 3 - next get the chroma collection
        # Step 4 - pass all the retrieved data to the main_workflow method
        # Step 5 - check if the user input is empty if true retun none
        # Step 6 - Else remove the paranthesis from the user input.
        # Step 7 - generate a summary for the user input the required format.
        # Step 8 - check if the summary is empty. if true retry the generation of the summary, you can do this until five times
        # (the above step is because we were getting errors while converting the generted summary to the json format.)
        # Step 9 - generate an sql query.
        # Step 10 - validate the generated query.
        # Step 11 - check if the query result is empty. if true then do a similarity search and retrieve the relevent info and return it.
        # Step 12 - else return the query result.

        
        logger.info(f"Executing NLQS query workflow for input: {user_input}")
        
        # Step 0 - Create the pre-requisite objects
        driver = self.connection_driver

        # Retrieve descriptions and types from db
        logger.debug("Retrieving column descriptions from vector database...")
        column_descriptions_dict = self.vectordb_driver.retrieve_descriptions_and_types_from_db()
        if column_descriptions_dict is None:
            logger.error("No data found in the database")
            raise ValueError("No data found in the database. Generate Column descriptions.")
        logger.debug("Column descriptions retrieved successfully")

        # Get the primary key for the table
        primary_key = driver.get_primary_key(self.table_name)
        logger.debug(f"Using primary key: {primary_key}")
        
        # Get Chroma Collection
        logger.debug("Getting Chroma collection...")
        chroma_data_collection = self.vectordb_driver.dataset_collection
        if chroma_data_collection is None:
            logger.error("Chroma Collection not found")
            raise ValueError("Chroma Collection not found in vectordb. Please create a collection.")
        logger.debug("Chroma collection retrieved successfully")

        # Step 5 - check if the user input is empty
        if not user_input.strip():
            logger.info("Empty user input received")
            return NLQSResult(records=[], uris=[])

        # Step 6 - Remove curly braces from input
        logger.debug("Processing user input...")
        user_input = re.sub(r"{|}", "", user_input)

        # Step 7 - Generate summary
        logger.debug("Generating input summary...")
        try:
            summarized_input = summarize(
                user_input=user_input,
                chat_history=chat_history,
                column_descriptions_dictionary=column_descriptions_dict["column_descriptions"],
                numerical_columns=column_descriptions_dict["numerical_columns"],
                categorical_columns=column_descriptions_dict["categorical_columns"],
                descriptive_columns=column_descriptions_dict["descriptive_columns"],
                llm=self.llm,
                vectordb=self.vectordb_driver,
            )
            logger.debug(f"Generated summary: {summarized_input}")
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}", exc_info=True)
            raise

        count = 0
        print(f"summarized_input: {summarized_input}")
        while not summarized_input.summary and count < 5:
            summarized_input = summarize(
                user_input=user_input,
                chat_history=chat_history,
                column_descriptions_dictionary=column_descriptions_dict["column_descriptions"],
                numerical_columns=column_descriptions_dict["numerical_columns"],
                categorical_columns=column_descriptions_dict["categorical_columns"],
                descriptive_columns=column_descriptions_dict["descriptive_columns"],
                llm=self.llm,
                vectordb=self.vectordb_driver,
            )
            count += 1
            if count == 5:
                raise ValueError("Unable to summarize the data.")

        intent = summarized_input.user_intent

        logger.info("--------------------------")
        logger.info(f"user input: {user_input}")
        logger.info(f"Summarized input: {summarized_input}")

        if intent == "sql_injection":
            # Kill the workflow if the user input is a SQL injection
            return NLQSResult(records=[], uris=[], is_input_irrelevant=True)
        elif intent == "phatic_communication":
            # Kill the workflow if the user input is phatic communication
            return NLQSResult(records=[], uris=[], is_input_irrelevant=True)

        # TODO: Figure out other intents in the future
        else:
            print("NLQS intent")

        # This is the standard workflow for the NLQS

        # TODO: We should reenable this        # # Check if the user requested columns exist
        # for column in summarized_input.user_requested_columns:
        #     if column not in column_descriptions:
        #         raise ValueError(f"Column {column} not found in the database.")

        print("checking for user requested columns...")
        if len(summarized_input.user_requested_columns) > 0:
            numerical_data = summarized_input.numerical_data
            categorical_data = summarized_input.categorical_data
            descriptive_data = summarized_input.descriptive_data
            identifier_data = summarized_input.identifier_data

            identifier_query_fragments = construct_quantitaive_search_query_fragments(identifier_data)
            quantitative_query_fragments = construct_quantitaive_search_query_fragments(numerical_data)
            categorical_query_fragments = construct_categorical_search_query_fragments(categorical_data)
            descriptive_query_fragments = construct_descriptive_search_query_fragments(
                descriptive_data, self.vectordb_driver
            )

            # Construct a search field that will capture all the data from the user input
            search_field_object = SearchField.construct_search_field(
                descriptive_query_fragments=[
                    fragment for fragments in descriptive_query_fragments.values() for fragment in fragments
                ],
                categorical_query_fragments=categorical_query_fragments,
                identifier_query_fragments=identifier_query_fragments,
                quantitative_query_fragments=quantitative_query_fragments,
                database_driver=self.connection_driver,
                database_name=DEFAULT_DB_NAME,
                table_name=DEFAULT_TABLE_NAME,
            )

            # Get all search results from the search field
            search_results = search_field_object.get_results()
            print(f"Search results: {search_results}")

            # Extract primary keys from search results
            all_primary_keys = []
            if "default" in search_results:
                for row in search_results["default"]:
                    if row and len(row) > 0:
                        # Assuming first column is primary key
                        all_primary_keys.append(row[0])

            # Remove duplicates while preserving order
            unique_primary_keys = list(dict.fromkeys(all_primary_keys))
            
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
                    result = NLQSResult(records=[], uris=[])
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
                    result = NLQSResult(records=records, uris=uris)

                logger.info(f"Query executed: {final_query}")
                logger.info(f"Found {len(records)} records")
                print(f"result: {result}")
        else:
            logger.info("No user requested columns found")
            result = NLQSResult(records=[], uris=[])

        return result

    def _get_connection_driver(self, connection_config: Union[SQLiteConnectionConfig, PostgresConnectionConfig]) -> SQLiteDriver | PostgresDriver:
        """Check if the connection configuration is valid."""
                # Initialize database connection
        if isinstance(connection_config, SQLiteConnectionConfig):
            logger.info("Using SQLite database")
            return SQLiteDriver(connection_config)
        elif isinstance(connection_config, PostgresConnectionConfig):
            logger.info("Using PostgreSQL database")
            return PostgresDriver(connection_config)
        else:
            logger.error("Invalid connection configuration")
            raise ValueError("Invalid connection configuration")
    
    def _iniitalize_llm(self):
        """Initialize the LLM object."""
        logger.debug("Initializing LLM...")
        # Create the LLM object
        llm = get_default_llm(use_azure=True)  # Use Azure OpenAI by default
        if llm is None:
            logger.error("Failed to initialize LLM")
            raise ValueError("Failed to initialize LLM")
        logger.info("LLM initialized")

        return llm
    
    def _initialize_embedding_model(self) -> AzureOpenAIEmbeddings:
        """Initialize the embedding model."""
        # Create the embedding model

        logger.info("LLM initialized")
        embedding_model = get_default_embedding_function(use_azure=True)
        if embedding_model is None:
            logger.error("Failed to initialize embedding model")
            raise ValueError("Failed to initialize embedding model")
        logger.info("LLM initialized")

        return embedding_model
    
    def _initialize_vectordb_driver(
        self,
        chroma_config: ChromaDBConfig
    ) -> VectorDBDriver:
        """Initialize the vector database driver."""
        logger.debug("Initializing vector database driver...")
        vectordb_driver = VectorDBDriver(
            chroma_config=chroma_config,
            embedding_function=self._initialize_embedding_model(),
        )
        if vectordb_driver is None:
            logger.error("Failed to initialize vector database driver")
            raise ValueError("Failed to initialize vector database driver")
        logger.info("Vector database driver initialized successfully")

        return vectordb_driver
    
    def _test_infrastructure(self):
        """Test if all required infrastructure components are available."""
        logger.debug("Checking ChromaDB collections...")
        if self.vectordb_driver.check_nlqs_collections_exists() is False:
            logger.error("ChromaDB collections do not exist")
            raise ValueError("ChromaDB collections do not exist. Please create them.")
        logger.info("ChromaDB collections verified")
    