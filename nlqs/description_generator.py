import json
import re
from typing import Dict, List, Optional, Union

import chromadb
import pandas as pd
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from nlqs.database.postgres import PostgresDriver
from nlqs.database.sqlite import SQLiteDriver
from nlqs.parameters import OPENAI_API_KEY
from utils.llm import get_default_llm
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# 1. pass the data in the databse
# 2. for each column in the table create a sample data for 5 non empty rows and remove '{|}'
# 3. pass the column name, it's data type and the sample data into the llm to generate desccriptions
# 4. in the instruction for llm, i passed some predifined descriptions to make the llm know how to write descriptions.
# these predifined descriptions will not effect any future changes..



def _get_prompt_for_description_generation():
    return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are an AI specialized in describing dataset columns. Your job is to analyze a given dataset column, 
                    including its name, datatype, and sample data, and generate a detailed description.

                    Guidelines:
                    - If the datatype is `int64` or `float64`, the column type is "numerical".
                    - If the datatype is `object`, the column type is either "categorical" or "descriptive" based on the data.
                    - Use the sample data to infer the potential meaning, importance, and use of the column.

                    data:
                    The following is a description of a dataset column:
                    Column Name: {column}
                    Data Type: {col_type}                    
                    Sample Data: {sample_data_str}

                    
                    For example:
                    "Product": "This column contains the name of the product. It is a text field and can be used for exact or partial matches.",
                    "Category": "This column contains the category of the product. It is a text field and can be used for exact or partial matches.",
                    "MedicalBenefits": "This column contains the medical benefits of the product. It is a text field and can be used for exact or partial matches.",
                    "CustomerRating": "This column contains the customer rating of the product. It is a numerical field and can be used for exact matches or range comparisons.",
                    "PurchaseFrequency": "This column contains the frequency of product purchase. It is a text field and can be used for exact or partial matches.",
                    "description": "This column contains the description of the product. It is a text field and can be used for exact or partial matches."
                    
                                        
                    Output format:
                    {{
                        "column_name": "{column}",
                        "description": "<Detailed description based on column meaning and sample data>",
                        "column_type": "<numerical, categorical, or descriptive>"
                    }}
                    Please provide a detailed description of this column, including its potential meaning, use, and importance in a dataset. Use sample data to identify the column's meaning.
                    """,
                ),
                ("user", "{user_input}"),
            ]
        )

def _validate_parsed_str(parsed_str: str) -> str:
    """
    Validates and cleans a JSON string to ensure it can be parsed correctly.
    This function attempts to fix common issues such as trailing commas and unescaped characters.

    Args:
        json_str (str): The JSON string to validate.
    Returns:
        str: A cleaned JSON string that can be parsed.
    """
    logger.debug(f"Validating parsed string: {parsed_str}")
    validated_str = None

    if parsed_str.startswith("```json"):
        logger.info("Parsed string starts with ```json, attempting to extract JSON block.")
        match = re.search(
            r"```json\n(.*?)\n```", parsed_str, re.DOTALL
        )  # The re.DOTALL flag allows the dot (.) to match newline characters as well.

        logger.debug(f"Match found: {match}")

        if match:
            validated_str = match.group(1)  # Extract the JSON content without the ```json and ``` markers

    return validated_str or parsed_str


def generate_column_descriptions(dataframe: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    logger.info("Generating column descriptions...")

    # Initialize an empty dictionary to store column descriptions and types
    descriptions = {}

    for column in dataframe.columns:
        # For each column, get data and create sample data from five non-empty rows, removing special characters.
        col_data = dataframe[column]
        col_type = col_data.dtype
        sample_data = dataframe[column].dropna().sample(min(5, len(dataframe[column]))).tolist()
        sample_data_str = ", ".join(map(str, sample_data))
        sample_data_str = re.sub("{|}", "", sample_data_str)

        # Build the chain for description generation
        prompt = _get_prompt_for_description_generation()
        llm = get_default_llm()
        output_parser = StrOutputParser() # TODO: See if JSONOutputParser can be used here
        chain = prompt | llm | output_parser

        # Generate the description
        response = chain.invoke(
            {
                "column": column,
                "col_type": col_type,
                "sample_data_str": sample_data_str,
                "user_input": "Please provide a detailed description of each column in the given dataset.",
            }
        )

        logger.info(f"Response: {response}")
        
        response = _validate_parsed_str(response)

        # Parse the JSON response
        json_response = json.loads(response)

        column_name = json_response["column_name"]
        description = json_response["description"]
        column_type = json_response["column_type"]

        # Store the description and type in the dictionary
        descriptions[column_name] = {
            "description": description,
            "column_type": column_type,
        }

    logger.info("Column descriptions generated successfully.")
    # Return the dictionary of column descriptions and types
    return descriptions # TODO: Create a dataclass for better structure and type safety


def store_descriptions_in_db(
    descriptions: Dict[str, Dict[str, str]],  # Updated to hold descriptions and types
    db_driver: Union[SQLiteDriver, PostgresDriver],
):
    # Create table to store column names, descriptions, and types in one table
    db_driver.execute_query(
        """
        CREATE TABLE IF NOT EXISTS column_metadata (
            column_name TEXT PRIMARY KEY,
            description TEXT,
            column_type TEXT
        )
        """
    )

    for column, metadata in descriptions.items():
        description = metadata["description"]
        column_type = metadata["column_type"]

        # Insert or replace each column's name, description, and type into the table
        db_driver.execute_query(
            f"""
            INSERT OR REPLACE INTO column_metadata (column_name, description, column_type)
            VALUES ("{column}", "{description}", "{column_type}")
            """
        )

    logger.info("Column metadata (column_name, description, column_type) stored in the database.")


def generate_and_store_column_description(df: pd.DataFrame, db_driver: Union[SQLiteDriver, PostgresDriver]):
    # Get column descriptions along with types
    column_descriptions = generate_column_descriptions(dataframe=df)

    # Store descriptions and column types in the database
    store_descriptions_in_db(
        descriptions=column_descriptions,
        db_driver=db_driver,
    )

    logger.debug(column_descriptions)
    logger.info("Column descriptions and column types stored in the database.")
