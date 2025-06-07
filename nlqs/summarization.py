import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, TypedDict, Union

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAI
from sqlalchemy import column

from nlqs.parameters import DEFAULT_DB_NAME, DEFAULT_TABLE_NAME
from nlqs.vectordb_driver import ColumnType, DataCollectionMetadata, VectorDBDriver
from utils.json_outputs import validate_llm_output_keys
from langchain_core.output_parsers import JsonOutputParser

# Create a logger object
logger = logging.getLogger(__name__)

# Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR)
logger.setLevel(logging.INFO)


class InputIntent(TypedDict):
    """Class to represent the input intent."""

    summary: str
    user_intent: str
    qualitative_statements: List[str]
    quantitative_statements: List[str]


@dataclass
class SummarizedInput:
    """Class to represent the summarized input."""

    summary: str
    numerical_data: Dict[str, str]
    categorical_data: Dict[str, str]
    descriptive_data: Dict[str, str]
    identifier_data: Dict[str, str]
    user_requested_columns: List[str]
    user_intent: str


REFERENCE_SUMMARIZED_INTENT_DICT = {
    "summary": "",
    "user_intent": "",
    "qualitative_statements": [],
    "quantitative_statements": [],
}


REFERENCE_SUMMARIZED_OUTPUT_DICT = {
    "quantitative_data": {},
    "qualitative_data": {},
    "user_requested_columns": [],
}


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
    categorical_columns: List[str],
    descriptive_columns: List[str],
    llm: Union[ChatOpenAI, OpenAI],
    vectordb: VectorDBDriver,
) -> SummarizedInput:
    """Summarizes the user input and returns the summary, quantitative data, and qualitative data, along with the user requested columns in a JSON format."""

    # Format the prompt template with column information
    available_columns = list(column_descriptions_dictionary.keys())
    column_descriptions = "\n".join([f"- {col}: {desc}" for col, desc in column_descriptions_dictionary.items()])
    
    prompt = f"""You are an expert at analyzing user queries and extracting structured information.

User request: {{input}}

Available database columns and their descriptions:
{column_descriptions}

Analyze the user request and extract:
1. Summary: Brief description of what the user wants
2. Numerical data: Extract any numerical filters/conditions (e.g., "age > 25", "price < 100")
3. Categorical data: Extract any categorical filters (e.g., "type = electronics", "status = active")
4. Descriptive data: Extract any text-based search terms or descriptions
5. User requested columns: Which columns the user wants to see in results
6. User intent: What the user wants to do (search, analyze, compare, etc.)

Return ONLY a JSON object with this structure:
{{
  "summary": "Brief description of request",
  "numerical_data": {{}}, 
  "categorical_data": {{}},
  "descriptive_data": {{}},
  "user_requested_columns": [],
  "user_intent": "search"
}}

Guidelines:
- For numerical_data: Use format like {{"column_name": "operator value"}} (e.g., {{"age": ">25", "price": "<100"}})
- For categorical_data: Use format like {{"column_name": "value"}} (e.g., {{"category": "electronics"}})
- For descriptive_data: Use format like {{"column_name": "search_term"}} for text searches
- For user_requested_columns: Include column names the user wants to see
- If no specific columns mentioned, leave user_requested_columns empty

Do not include any other text or formatting, just the JSON object."""
    
    intent_classification_prompt = ChatPromptTemplate.from_template(prompt)

    output_parser = JsonOutputParser()
    chain = intent_classification_prompt | llm | output_parser
    
    # Get the parsed JSON output directly
    data_dict = chain.invoke({"input": user_input})
    
    print(f"parsed output : {data_dict} and is type {type(data_dict)}")
    
    # Handle the case where JsonOutputParser returns a string instead of dict
    if isinstance(data_dict, str):
        try:
            data_dict = json.loads(data_dict)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON string: {str(e)}")
            data_dict = {}
    
    # Ensure we have all required fields with defaults
    required_fields = {
        "summary": user_input,
        "numerical_data": {},
        "categorical_data": {},
        "descriptive_data": {},
        "user_requested_columns": [],
        "user_intent": "search"
    }
    
    # Fill in missing fields with defaults
    for field, default in required_fields.items():
        if field not in data_dict or data_dict[field] is None:
            data_dict[field] = default
            
    return SummarizedInput(
        summary=data_dict["summary"],
        numerical_data=data_dict["numerical_data"],
        categorical_data=data_dict["categorical_data"],
        descriptive_data=data_dict["descriptive_data"],
        identifier_data={},  # This field is not used but required by the class
        user_requested_columns=data_dict["user_requested_columns"],
        user_intent=data_dict["user_intent"]
    )


def get_validated_user_requested_columns(
    vectordb_driver: VectorDBDriver, summazied_user_requested_columns: List[str], table_name: str, db_name: str
) -> List[str]:
    """Validates the user requested columns and returns a list of valid columns.

    Args:
        summazied_user_requested_columns (List[str]): The user requested columns.
        table_name (str): The table name.
        db_name (str): The database name.

    Returns:
        List[str]: A list of valid user requested columns.
    """
    if not summazied_user_requested_columns:
        return []
    if len(summazied_user_requested_columns) < 1:
        return []

    ret = []
    # Go through each of the columns
    for column_name in summazied_user_requested_columns:
        # Check if the columns is present in the data
        exists = vectordb_driver.check_if_column_name_exists(column_name, table_name, db_name)

        # If it exists, add it to the list, continue to the next column
        if exists:
            ret.append(column_name)
            continue

        # If the column does not exist, find the closest column name
        closest_column_name, column_type = vectordb_driver.get_closest_column_from_description(
            approximate_column_name=column_name,
            users_description="",
            sample_data_strings=[],
            database_name=db_name,
            table_name=table_name,
        )

        # If the closest column name is not found, print a warning and continue to the next column
        if not closest_column_name:
            logger.warning(f"Closest column name not found for column '{column_name}'")
            continue

        # Now add the closest column name to the list
        ret.append(closest_column_name)

    return ret


# def qualitaive_search(collection: chromadb.Collection, data: Dict[str, str], primary_key: str) -> List[str]:
#     """Performs a similarity search on the database and returns all similar results.

#     Args:
#         collection (chromadb.Collection): The ChromaDB collection to search.
#         data (Dict[str, str]): A dictionary of qualitative data to search for.
#         primary_key (str): The primary key column name in the database.

#     Returns:
#         List[str]: A dictionary containing the search results.
#     """
#     all_ids = []

#     for column, condition in data.items():
#         query_result = collection.query(query_texts=condition, n_results=10, where={"column_name": column})

#         if query_result:
#             ids_for_column = set()  # Use a set to store unique IDs for this column
#             for result in query_result["metadatas"]:
#                 for item in result:
#                     id_value = item.get(primary_key)
#                     if id_value is not None:
#                         ids_for_column.add(str(id_value))  # Convert to string for comparison
#             all_ids.append(ids_for_column)

#     # Find the intersection of IDs across all columns
#     common_ids = set.intersection(*all_ids) if all_ids else set()

#     return list(common_ids)
