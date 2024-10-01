import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union
import chromadb
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAI
from nlqs.database.postgres import PostgresDriver
from nlqs.database.sqlite import SQLiteDriver

# Create a logger object
logger = logging.getLogger(__name__)

# Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR)
logger.setLevel(logging.INFO)


@dataclass
class SummarizedInput:
    """Class to represent the summarized input."""

    summary: str
    numerical_data: Dict[str, str]
    categorical_data: Dict[str, str]
    descriptive_data: Dict[str, str]
    numerical_data: Dict[str, str]
    categorical_data: Dict[str, str]
    descriptive_data: Dict[str, str]
    user_requested_columns: List[str]
    user_intent: str


# Default system prompt for the LLM.
DEFAULT_SYSTEM_PROMPT = (
    "You are a professional medical assistant, adept at handling inquiries related to medical products."
)


# Generates a prompt for the LLM based on the instruction and system prompt.
def get_prompt(instruction: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    """Generates the prompt for the LLM.

    Args:
        instruction (str): The instruction for the LLM.
        system_prompt (str, optional): The system prompt for the LLM. Defaults to DEFAULT_SYSTEM_PROMPT.

    Returns:
        str: The prompt for the LLM.

    """
    SYSTEM_PROMPT = f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
    return f"[INST]{SYSTEM_PROMPT}{instruction}[/INST]"


# Function to identify qualitative and quantitative data and user intent
def summarize(
    user_input: str,
    chat_history: List[Tuple[str, str]],
    column_descriptions_dictionary: Dict[str, str],
    numerical_columns: List[str],
    descriptive_columns: List[str],
    categorical_columns: List[str],
    llm: Union[ChatOpenAI, OpenAI],
) -> SummarizedInput:
    """Summarizes the user input and returns the summary, quantitative data, and qualitative data, along with the user requested columns in a JSON format.

    Args:
        user_input (str): The user input.
        chat_history (list[(str, str)]): The chat history.
        column_descriptions (dict[str, str]): The column descriptions.
        numerical_columns (list[str]): The numerical columns.
        descriptive_columns (list[str]): The descriptive columns.
        categorical_columns (list[str]): The categorical columns.
        llm (Union[ChatOpenAI, OpenAI]): The LLM object.(Contains the details of the language we are using.)

    Returns:
        dict: {
            "summary": str,
            "numerical_data": {
                "column name : str" : "Data mentioned about that column by the user : str",
                "column name : str" : "Data mentioned about that column by the user : str",
                "column name : str" : "Data mentioned about that column by the user : str",
            },
            "descriptive_data": {
                "column name : str" : "Data mentioned about that column by the user : str",
                "column name : str" : "Data mentioned about that column by the user : str",
                "column name : str" : "Data mentioned about that column by the user : str",
            },
            "categorical_data": {
                "column name : str" : "Data mentioned about that column by the user : str",
                "column name : str" : "Data mentioned about that column by the user : str",
                "column name : str" : "Data mentioned about that column by the user : str",
            },
            "user_requested_columns": list,
            "user_intent":str,
        }
    """

    column_descriptions = list(column_descriptions_dictionary.items())

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
                You will receive a user input and the chat history. Your task is to:
                
                1. **Single-Word Queries**: If the user input is a single word or very short (e.g., one or two words), provide a direct response if possible. If the query is unclear, prompt the user to elaborate.
                - Example response: "It seems you're asking about something specific. Could you provide more details?"

                2. **Structured Analysis**: For all other inputs, analyze the user input and identify key details based on our available data and chat history.
                
                3. Summarize the input, classifying the data into categorical_data, descriptive_data and categorical_data categories.
                
                4. Identify relevant columns from which we can provide an answer. Pay close attention to the user's intent and specific mentions of data columns:
                - Are they seeking information about products, medications, treatments, or other relevant categories?
                - If the user is seeking information about a product, also provide the URL of the product if available.
                - Look for explicit mentions of column names, synonyms, or phrases that indicate the type of information requested. If the user specifies certain attributes or metrics, consider these as user-requested columns.
                - Relevant data is provided, use that data to classify the columns.

                5. Classify the user's intent. Possible intents include: phatic_communication, sql_injection, profanity, and other.

                6. Output the result in a JSON format.

                7. Do not output any other information except the JSON. Do not add [OUT], [/OUT] to the output.(!important)
                
                The output JSON should have the following structure:
                `
                    "summary": "summary of the user input",
                    "numerical_data":
                    "numerical_data":
                                        ` 
                                        "column name": "Data mentioned about that column by the user. Example- < 4",
                                        "column name": "Data mentioned about that column by the user. Example- > 6.215",
                                        "column name": "Data mentioned about that column by the user. Example- >= 3.14 or <= 2.718",
                                        `,
                    "categorical_data": 
                                        ` 
                                        "column name": "Data mentioned about that column by the user",
                                        "column name": "Data mentioned about that column by the user",
                                        "column name": "Data mentioned about that column by the user",
                                        `,
                    "categorical_data": 
                                        ` 
                                        "column name": "Data mentioned about that column by the user",
                                        "column name": "Data mentioned about that column by the user",
                                        "column name": "Data mentioned about that column by the user",
                                        `,
                    "descriptive_data":
                                        ` 
                                        "column name": "Data mentioned about that column by the user",
                                        "column name": "Data mentioned about that column by the user",
                                        "column name": "Data mentioned about that column by the user",
                                        `,
                    "user_requested_columns": "List of columns the user wants data from. If none, leave it as an empty list. Always add product and url to this column.",
                    "user_intent": "The user's intent. If none, leave it as an empty string.",
                `
                
                The data we have and chat history:
                Data:{column_descriptions}\n\n 
                numerical columns in the data: {numerical_columns}\n\n 
                descriptive columns in the data: {descriptive_columns}\n\n 
                categorical columns in the data: {categorical_columns}\n\n
                chat history: {chat_history}

                Now, summarize the user input, chat history and provide the structured output in JSON format.
                """,
            ),
            ("human", f"{user_input}"),
        ]
    )

    # print(f"prompt: {prompt}")
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    summarized_input_str = str(chain.invoke({"user_input": user_input}))

    print(f"summarized_input_str: {summarized_input_str}")

    print("------------------------------------------------------------------------")

    try:
        # Attempt to parse the summarized input as JSON
        summarized_input_dict = json.loads(summarized_input_str)
    except json.JSONDecodeError:
        # If parsing fails, return an empty SummarizedInput
        summarized_input_dict = {}

    logger.info("--------------------------")
    logger.info(f"user input: {user_input}")
    logger.info(f"Summarized input: {summarized_input_dict}")

    summarized_input = SummarizedInput(
        summary=summarized_input_dict.get("summary", ""),
        numerical_data=summarized_input_dict.get("numerical_data", {}),
        categorical_data=summarized_input_dict.get("categorical_data", {}),
        descriptive_data=summarized_input_dict.get("descriptive_data", {}),
        user_requested_columns=summarized_input_dict.get("user_requested_columns", []),
        user_intent=summarized_input_dict.get("user_intent", ""),
    )

    return summarized_input


def columns_chroma_lookup(
    chroma_client, numerical_data: Dict[str, str], categorical_data: Dict[str, str], descriptive_data: Dict[str, str]
):
    collection = chroma_client.get_collection(name="column_info")
    for item, value in numerical_data:
        column = collection.query(query_texts=[value], n_results=1).get("documents")[0][0]
        # numerical_data[item] = column
        print(f"original column: {item}")
        print(f"column: {column}")

    pass


def generate_numerical_search_query(quantitative_data: Dict[str, str], table_name: str, primary_key: str) -> str:
    if not quantitative_data:
        return ""  # Return an empty string if the dictionary is empty

    query_parts = []
    for column, condition in quantitative_data.items():
        # Handle different comparison operators
        if "<" in condition:
            operator = "<"
        elif ">" in condition:
            operator = ">"
        elif "<=" in condition:
            operator = "<="
        elif ">=" in condition:
            operator = ">="
        elif "=" in condition:
            operator = "="
        else:
            operator = "LIKE"  # Default to LIKE for other conditions

        # Extract the value from the condition
        value = condition.replace(operator, "").strip()

        # Handle quoting for string values
        if not value.isdigit():  # Add single quotes for non-numeric values
            value = f"'{value}'"

        # Construct the query part with column name in double quotes
        query_part = f'"{column}" {operator} {value}'
        query_parts.append(query_part)

    # Combine the query parts with AND
    query_constraints = " AND ".join(query_parts)

    # Construct the final query with table and primary key also in double quotes
    query = f'SELECT "{primary_key}" FROM "{table_name}" WHERE {query_constraints}'
    return query


def descriptive_search(collection: chromadb.Collection, data: Dict[str, str], primary_key: str) -> List[int]:
    """Performs a similarity search on the database and returns up to 5 similar results per column.

    Args:
        collection (chromadb.Collection): The ChromaDB collection to search.
        data (Dict[str, str]): A dictionary of qualitative data to search for.
        primary_key (str): The primary key column name in the database.

    Returns:
        List[int]: A list of unique IDs from the search results.
    """
    ids_per_column = {}

    for column, condition in data.items():
        query_result: chromadb.QueryResult = collection.query(
            query_texts=condition, n_results=5, where={"column_name": column}
        )

        if query_result["metadatas"]:
            ids_for_column = set()
            for result in query_result["metadatas"]:
                for item in result:
                    id_value = item.get(primary_key)
                    if id_value is not None:
                        ids_for_column.add(int(id_value))
            ids_per_column[column] = list(ids_for_column)

    # print(f"ids_per_column: {ids_per_column}")

    # Flatten the list of lists into a single list of unique IDs
    all_ids = list(set([id_val for sublist in ids_per_column.values() for id_val in sublist]))
    return all_ids


def categorical_search(
    collection: chromadb.Collection,
    data: Dict[str, str],
    db_driver: Union[SQLiteDriver, PostgresDriver],
    primary_key: str,
) -> List[int]:
    """Performs a similarity search on the database and returns up to 5 similar results per column.

    Args:
        collection (chromadb.Collection): The ChromaDB collection to search.
        data (Dict[str, str]): A dictionary of qualitative data to search for.
        db_driver (Union[SQLiteDriver, PostgresDriver]): database driver.
        primary_key (str): The primary key column name in the database.

    Returns:
        List[int]: A list of unique IDs from the search results.
    """
    ids_per_column = {}

    for column, condition in data.items():
        query_result = collection.query(query_texts=[condition], n_results=1, where={"column_name": column})

        print(f"Query result: {query_result["documents"]}")


        if query_result["documents"]:
            ids_for_column = set()
            print(f"Query result: {query_result["documents"][0]}")  

            # Extract the string directly
            query_value = query_result["documents"][0][0]  # Get the first element of the list

            # Use parameter binding
            query = (
                f'SELECT {primary_key} FROM {db_driver.db_config.dataset_table_name} WHERE "{column}" = \'{query_value}\''
            )
            ids_for_column_uncleaned = db_driver.execute_query(query)

            if ids_for_column_uncleaned:
                ids_for_column = {item[0] for item in ids_for_column_uncleaned}

            ids_per_column[column] = list(ids_for_column)

    # print(f"ids_per_column: {ids_per_column}")

    # Flatten the list of lists into a single list of unique IDs
    all_ids = list(set([id_val for sublist in ids_per_column.values() for id_val in sublist]))
    return all_ids


def get_chroma_collection(collection_name: str, client) -> Tuple[chromadb.Collection, chromadb.Collection]:
    """Retrieves data from the chroma collection, if there is no chroma collection it creates one.

    Args:
        collection_name (str): name of chroma collection for descriptive data.
        client (_type_): chroma client

    Returns:
        Tuple[chromadb.Collection, chromadb.Collection]: A tuple containing the chroma collection for descriptive data and the chroma collection for categorical data.
    """

    collections = [col.name for col in client.list_collections()]

    print(f"collections: {collections}")
    categorical_collection_name = "categorical_data"  # Default name for categorical collection

    # Descriptive Data Collection
    if collection_name in collections and categorical_collection_name in collections:
        print(f"Collection '{collection_name}' already exists, getting existing collection...")
        descriptive_collection = client.get_collection(collection_name)
        categorical_collection = client.get_collection(categorical_collection_name)
    else:
        print(f"Collection '{collection_name}' does not exists, Create a collection...")

        raise ValueError("Chroma collection doesn't exist. Create a chroma collection!!")

    return descriptive_collection, categorical_collection
