import json
import re
from typing import Dict, List, Optional, Union
import chromadb
import pandas as pd
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic.v1 import SecretStr
from pathlib import Path
from nlqs.database.postgres import PostgresDriver
from nlqs.database.sqlite import SQLiteDriver
from nlqs.nlqs import ChromaDBConfig
from nlqs.parameters import OPENAI_API_KEY
from nlqs.database.postgres import PostgresConnectionConfig
from nlqs.database.sqlite import SQLiteConnectionConfig
from scripts.parameters import (
    CHROMA_COLLECTION_NAME,
    OUTPUT_COLUMNS,
    SQL_TABLE_NAME,
    SQLITE_DB_FILE,
    SQL_TABLE_NAME,
    SUPABASE_DATABASE_NAME,
    SUPABASE_HOST,
    SUPABASE_PASSWORD,
    SUPABASE_PORT,
    SUPABASE_USER,
    URL_COLUMN,
    VECTORDB_HOST,
    VECTORDB_PORT,
)


# 1. pass the data in the databse
# 2. for each column in the table create a sample data for 5 non empty rows and remove '{|}'
# 3. pass the column name, it's data type and the sample data into the llm to generate desccriptions
# 4. in the instruction for llm, i passed some predifined descriptions to make the llm know how to write descriptions.
# these predifined descriptions will not effect any future changes..
def generate_column_descriptions(dataframe: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    print("Generating column descriptions...")

    # Initialize an empty dictionary to store column descriptions and types
    descriptions = {}

    for column in dataframe.columns:
        # For each column, get data and create sample data from five non-empty rows, removing special characters.
        col_data = dataframe[column]
        col_type = col_data.dtype
        sample_data = dataframe[column].dropna().sample(min(5, len(dataframe[column]))).tolist()
        sample_data_str = ", ".join(map(str, sample_data))
        sample_data_str = re.sub("{|}", "", sample_data_str)

        response = None

        # Prepare the prompt for LLM
        prompt = ChatPromptTemplate.from_messages(
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
                    "Product": "This column contains the name of the product. It is a text field and can be used for exact or partial matches.", "column_type": "descriptive"
                    "Category": "This column contains the category of the product. It is a text field and can be used for exact or partial matches.", "column_type": "categorical"
                    "MedicalBenefits": "This column contains the medical benefits of the product. It is a text field and can be used for exact or partial matches.", "column_type": "descriptive"
                    "CustomerRating": "This column contains the customer rating of the product. It is a numerical field and can be used for exact matches or range comparisons.", "column_type": "numerical"
                    "PurchaseFrequency": "This column contains the frequency of product purchase. It is a text field and can be used for exact or partial matches.", "column_type": "categorical"
                    "description": "This column contains the description of the product. It is a text field and can be used for exact or partial matches.", "column_type": "descriptive"
                    
                                        
                    Output format:
                    {{
                        "column_name": "{column}",
                        "description": "<Detailed description based on column meaning and sample data>",
                        "column_type": "<numerical, categorical, or descriptive>"
                    }}
                    Please provide a detailed description of this column, including its potential meaning, use, and importance in a dataset. Use sample data to identify the column's meaning.
                    Also please do not anything extra other than the output format.
                    """,
                ),
                ("user", "{user_input}"),
            ]
        )

        llm = ChatOpenAI(
            model="gpt-4",
            api_key=SecretStr(OPENAI_API_KEY),
            temperature=0.0,
            verbose=True,
        )

        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser

        # Generate the description
        response = chain.invoke(
            {
                "column": column,
                "col_type": col_type,
                "sample_data_str": sample_data_str,
                "user_input": "Please provide a detailed description of the column in the given dataset using the specified format. Additionally, include sample data in the description.",
            }
        )

        print(f"response: {response}")

        if response.startswith("```json"):
            # Extract JSON from the response (apply this consistently)
            json_match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
            if json_match:
                response = json_match.group(1)
            else:
                print(f"Warning: No JSON found in response for column '{column}'")
                # Handle the case where no JSON is found (e.g., skip the column)
                continue

        json_response = json.loads(response)

        column_name = json_response["column_name"]
        description = json_response["description"]
        column_type = json_response["column_type"]

        # Store the description and type in the dictionary
        descriptions[column_name] = {
            "description": description,
            "column_type": column_type,
        }

    # Return the dictionary of column descriptions and types
    return descriptions


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
        description = metadata.get("description")
        column_type = metadata.get("column_type")

        # Log the values to ensure they are correct
        print(f"Inserting column: {column}, description: {description}, type: {column_type}")

        # Check if all required fields are present
        if not column or not description or not column_type:
            print(f"Skipping column {column} due to missing data")
            continue

        if isinstance(db_driver, SQLiteDriver):
            # SQLite syntax for inserting or replacing records
            query = f"""
                INSERT OR REPLACE INTO column_metadata (column_name, description, column_type)
                VALUES ('{column}', '{description.replace("'", "''")}', '{column_type}')
            """
        elif isinstance(db_driver, PostgresDriver):
            # Postgres syntax for inserting with ON CONFLICT clause
            query = f"""
                INSERT INTO column_metadata (column_name, description, column_type)
                VALUES ('{column}', '{description.replace("'", "''")}', '{column_type}')
                ON CONFLICT (column_name) DO UPDATE SET
                    description = EXCLUDED.description,
                    column_type = EXCLUDED.column_type;
            """
        else:
            raise ValueError("Unsupported database driver type")

        # Error handling for query execution
        try:
            print(f"Executing query: {query}")  # Log the full query
            db_driver.execute_query(query)
        except Exception as e:
            print(f"Error inserting column {column}: {e}")

    print("Column metadata (name, description, type) stored in the database.")


def generate_store_column_description(df: pd.DataFrame, db_driver: Union[SQLiteDriver, PostgresDriver], chroma_client):

    # Get column descriptions along with types
    column_descriptions = generate_column_descriptions(dataframe=df)

    # Store descriptions and column types in the database
    store_descriptions_in_db(
        descriptions=column_descriptions,
        db_driver=db_driver,
    )

    print(column_descriptions)
    print("Column descriptions and column types stored in the database.")

    collection = chroma_client.create_collection("column_info")

    for column_name, metadata in column_descriptions.items():
        collection.add(
            documents=[metadata["description"]], ids=[column_name], metadatas=[{"column_type": metadata["column_type"]}]
        )


if __name__ == "__main__":
    # SQLite configuration
    # connection_config = SQLiteConnectionConfig(
    #     db_file=Path(SQLITE_DB_FILE), dataset_table_name=SQL_TABLE_NAME, uri_column="URL", output_columns=OUTPUT_COLUMNS
    # )

    # Postgres configuration
    connection_config = PostgresConnectionConfig(
        host=SUPABASE_HOST,
        port=int(SUPABASE_PORT),
        user=SUPABASE_USER,
        password=SUPABASE_PASSWORD,
        database_name=SUPABASE_DATABASE_NAME,
        dataset_table_name=SQL_TABLE_NAME,
        uri_column=URL_COLUMN,
    )

    connection_driver = None

    if isinstance(connection_config, SQLiteConnectionConfig):
        connection_driver = SQLiteDriver(connection_config)
    elif isinstance(connection_config, PostgresConnectionConfig):
        connection_driver = PostgresDriver(connection_config)
    elif connection_driver is None:
        raise ValueError("Initialize or enter connection config..")

    connection_driver.connect()

    # ChromaDB configuration
    # chroma_config = ChromaDBConfig(collection_name=CHROMA_COLLECTION_NAME)  # local chroma

    # remote config
    chroma_config = ChromaDBConfig(
        collection_name=CHROMA_COLLECTION_NAME, is_local=False, host=VECTORDB_HOST, port=int(VECTORDB_PORT)
    )

    chroma_type = chroma_config.is_local
    if chroma_type:
        chroma_client = chromadb.PersistentClient(path=str(chroma_config.persist_path))
    else:
        chroma_client = chromadb.HttpClient(port=chroma_config.port, host=chroma_config.host)

    df = connection_driver.fetch_data_from_database(table_name=connection_config.dataset_table_name)
    generate_store_column_description(df=df, db_driver=connection_driver, chroma_client=chroma_client)
