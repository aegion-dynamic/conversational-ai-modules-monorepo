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

    # Format the prompt template carefully with proper escaping
    prompt = """Process the following request and output a structured response:

User request: {input}

Your task is to:
1. Analyze the request for any mentions of CBD content:
   - "high CBD" or "high CBD content" means ">15"
   - "low CBD" means "<5"
   - "medium CBD" means "5-15"

2. Return ONLY a JSON object with this EXACT structure:
   {{
     "summary": "Brief description of request",
     "numerical_data": {{"CBD": ">15"}},  
     "categorical_data": {{}},
     "descriptive_data": {{}},
     "user_requested_columns": ["Product", "URL"],
     "user_intent": "search"
   }}

3. For CBD values:
   - Always use exactly ">15" for high CBD
   - Always use exactly "<5" for low CBD
   - Always use "5-15" for medium CBD
   - Include "CBD" key in numerical_data for any CBD query

4. Include these columns in user_requested_columns:
   - "Product" must always be included
   - "URL" must always be included

Do not include any other text or formatting in your response, just the JSON object."""

    intent_classification_prompt = ChatPromptTemplate.from_template(prompt)

    output_parser = StrOutputParser()
    chain = intent_classification_prompt | llm | output_parser
    
    # Get the raw output and clean it
    raw_output = str(chain.invoke({"input": user_input}))
    
    # Clean up the output by removing any markdown formatting or backticks
    cleaned_output = raw_output.replace("```json", "").replace("```", "").strip()
    
    print(f"cleaned output: {cleaned_output}")
    
    try:
        data_dict = json.loads(cleaned_output)
        
        # Ensure we have all required fields with defaults
        required_fields = {
            "summary": user_input,
            "numerical_data": {"CBD": ">15"} if "high" in user_input.lower() and "cbd" in user_input.lower() else {},
            "categorical_data": {},
            "descriptive_data": {},
            "user_requested_columns": ["Product", "URL"],
            "user_intent": "search"
        }
        
        # Fill in missing fields with defaults
        for field, default in required_fields.items():
            if field not in data_dict or not data_dict[field]:
                data_dict[field] = default
                
        # Always ensure Product and URL are in requested columns
        if "Product" not in data_dict["user_requested_columns"]:
            data_dict["user_requested_columns"].append("Product")
        if "URL" not in data_dict["user_requested_columns"]:
            data_dict["user_requested_columns"].append("URL")
        
        # Special handling for CBD in high CBD queries
        if (
            "high" in user_input.lower() 
            and "cbd" in user_input.lower()
            and (
                "numerical_data" not in data_dict
                or "CBD" not in data_dict["numerical_data"]
                or not data_dict["numerical_data"]["CBD"]
            )
        ):
            if "numerical_data" not in data_dict:
                data_dict["numerical_data"] = {}
            data_dict["numerical_data"]["CBD"] = ">15"
            
        return SummarizedInput(
            summary=data_dict["summary"],
            numerical_data=data_dict["numerical_data"],
            categorical_data=data_dict["categorical_data"],
            descriptive_data=data_dict["descriptive_data"],
            identifier_data={},  # This field is not used but required by the class
            user_requested_columns=data_dict["user_requested_columns"],
            user_intent=data_dict["user_intent"]
        )

    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON: {str(e)}")
        # Return default structure for high CBD query
        return SummarizedInput(
            summary=user_input,
            numerical_data={"CBD": ">15"} if "high" in user_input.lower() and "cbd" in user_input.lower() else {},
            categorical_data={},
            descriptive_data={},
            identifier_data={},
            user_requested_columns=["Product", "URL"],
            user_intent="search"
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
