import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import chromadb
from langchain_openai import ChatOpenAI
from pydantic.v1 import SecretStr
from discord_bot.parameters import LOGGER_FILE
from nlqs.database.postgres import PostgresConnectionConfig, PostgresDriver
from nlqs.database.sqlite import SQLiteConnectionConfig, SQLiteDriver
from nlqs.parameters import OPENAI_API_KEY
from nlqs.query import (
    categorical_search,
    generate_numerical_serach_query,
    descriptive_search,
    summarize,
    get_chroma_collection,
)

# Create a logger object
logger = logging.getLogger(__name__)

# Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR)
logger.setLevel(logging.INFO)

# Create a file handler to save logs
file_handler = logging.FileHandler(LOGGER_FILE)

# Create a formatter to format the log messages
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Add the formatter to the file handler
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)


@dataclass
class ChromaDBConfig:
    collection_name: str
    persist_path: Path = Path("../chroma")
    host: str = "localhost"
    port: int = 8000
    is_local: bool = True


@dataclass
class NLQSResult:
    records: List[Dict[str, Any]]
    uris: List[str]


class NLQS:
    def __init__(
        self, connection_config: Union[SQLiteConnectionConfig, PostgresConnectionConfig], chroma_config: ChromaDBConfig
    ) -> None:
        # TODO - Figure out what the constructor parameters are
        if isinstance(connection_config, SQLiteConnectionConfig):
            self.connection_driver = SQLiteDriver(connection_config)
        elif isinstance(connection_config, PostgresConnectionConfig):
            self.connection_driver = PostgresDriver(connection_config)

        else:
            raise ValueError("Invalid connection configuration")

        # Initialize the connection to the database
        self.connection_driver.connect()

        # Create the llm object
        # Initializes the ChatOpenAI LLM model
        self.llm = ChatOpenAI(temperature=0, model="gpt-4-turbo", api_key=SecretStr(OPENAI_API_KEY), max_tokens=1000)

        self.chroma_config = chroma_config
        chroma_type = chroma_config.is_local
        if chroma_type:
            self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_config.persist_path))
        else:
            self.chroma_client = chromadb.HttpClient(port=chroma_config.port, host=chroma_config.host)

        self.table_name = connection_config.dataset_table_name
        self.uri_column = connection_config.uri_column
        self.output_columns = connection_config.output_columns

        # TODO - Figure out if we need to create introspection table, and create
        pass

    # Step 4
    def execute_nlqs_workflow(self, user_input: str, chat_history: List[Tuple[str, str]]) -> NLQSResult:
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

        # Step 0 - Create the pre-requisite objects

        # Step 5
        if not user_input.strip():
            result = NLQSResult(records=[], uris=[])
            return result

        # Database Connection
        driver = self.connection_driver

        # TODO: We should not retrieve column descriptions and chroma collections for every user query.
        column_descriptions, numerical_columns, categorical_columns, descriptive_columns = (
            driver.retrieve_descriptions_and_types_from_db()
        )

        if column_descriptions == {}:
            raise ValueError("No column descriptions found in the database. Generate Column descriptions.")

        primary_key = driver.get_primary_key(self.table_name)

        # Chroma Collection: descriptive_collection, categorical_collection
        descriptive_collection, categorical_collection = get_chroma_collection(
            collection_name=self.chroma_config.collection_name, client=self.chroma_client
        )

        # Step 6
        user_input = re.sub(r"{|}", "", user_input)

        # Step 7
        summarized_input = summarize(
            user_input=user_input,
            chat_history=chat_history,
            column_descriptions_dictionary=column_descriptions,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            descriptive_columns=descriptive_columns,
            llm=self.llm,
        )

        count = 0
        print(f"summarized_input: {summarized_input}")
        while not summarized_input.summary and count < 5:
            summarized_input = summarize(
                user_input=user_input,
                chat_history=chat_history,
                column_descriptions_dictionary=column_descriptions,
                numerical_columns=numerical_columns,
                categorical_columns=categorical_columns,
                descriptive_columns=descriptive_columns,
                llm=self.llm,
            )
            count += 1
            if count == 5:
                result = ["Summarization failed. Please try again."]
                break

        intent = summarized_input.user_intent

        logger.info("--------------------------")
        logger.info(f"user input: {user_input}")
        logger.info(f"Summarized input: {summarized_input}")

        if intent == "sql_injection":
            result = NLQSResult(records=[], uris=[])

        else:
            print("checking for user requested columns...")
            if summarized_input.user_requested_columns:
                numerical_data = summarized_input.numerical_data
                categorical_data = summarized_input.categorical_data
                descriptive_data = summarized_input.descriptive_data

                numerical_query = generate_numerical_serach_query(numerical_data, self.table_name, primary_key)
                numerical_ids_uncleaned = driver.execute_query(numerical_query)

                numerical_ids = []

                if numerical_ids_uncleaned:
                    numerical_ids = [item[0] for item in numerical_ids_uncleaned]
                    print(f"numerical_ids: {numerical_ids}")

                categorical_ids = categorical_search(categorical_collection, categorical_data, driver, primary_key)
                print(f"categorical_ids: {categorical_ids}")

                descriptive_ids = descriptive_search(descriptive_collection, descriptive_data, primary_key)
                print(f"descriptive_ids: {descriptive_ids}")

                # Find the intersection of quantitative_ids and qualitative_ids
                if not numerical_ids or not descriptive_ids or not categorical_ids:
                    intersection_ids = numerical_ids or descriptive_ids or categorical_ids
                else:
                    intersection_ids = list(set(numerical_ids) & set(descriptive_ids) & set(categorical_ids))

                # Ensure intersection_ids is set to qualitative_ids if it's empty
                if not intersection_ids:
                    intersection_ids = descriptive_ids

                print(f"intersection_ids: {intersection_ids}")

                # Initial query to retrieve all columns based on the intersection IDs
                final_query = f"SELECT * FROM {self.table_name} WHERE {primary_key} IN ({','.join(str(id) for id in intersection_ids)})"

                # Get the columns in the order they appear in the database
                columns_database = driver.get_database_columns(self.table_name)

                # Variables for specific columns
                uri_column = self.uri_column
                output_columns = self.output_columns

                # If output_columns is specified, modify the query to select only those columns
                if output_columns:
                    final_query = f"SELECT {','.join(col for col in output_columns)} FROM {self.table_name} WHERE {primary_key} IN ({','.join(str(id) for id in intersection_ids)})"
                    data_retreived = driver.execute_query(final_query)
                    # Since we now have a subset of columns, use output_columns directly
                    columns_to_use = output_columns
                else:
                    # Execute the query to retrieve the data with all columns
                    data_retreived = driver.execute_query(final_query)
                    columns_to_use = columns_database

                # Initialize lists to hold records and URIs
                records = []
                uris = []

                if not data_retreived:
                    result = NLQSResult(records=[], uris=[])
                    return result

                # Process the retrieved data
                for row in data_retreived:
                    record = dict(zip(columns_to_use, row))
                    if uri_column in record:
                        uris.append(str(record[uri_column]))
                        del record[uri_column]  # Remove the URI column data from the record
                    records.append(record)

                # Create the result object
                result = NLQSResult(records=records, uris=uris)

                print(f"result: {result}")
                logger.info(f"result: {result}")
            else:
                result = NLQSResult(records=[], uris=[])

        return result
