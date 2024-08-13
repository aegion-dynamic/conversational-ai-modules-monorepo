from typing import List, Optional, Union
import chromadb
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic.v1 import SecretStr

from nlqs.parameters import OPENAI_API_KEY
from nlqs.database.sqlite import SQLiteDriver
from nlqs.database.postgres import PostgresDriver
from pathlib import Path
from dataclasses import dataclass
import pandas as pd

@dataclass
class ChromaDBConfig:
    collection_name: str
    persist_path: Path
    is_local: bool = True


# TODO - Use the database Object for doing this
def get_column_descriptions(dataframe) -> dict:
    """Get column descriptions from OpenAI API."""
    # Initialize an empty dictionary to store column descriptions
    descriptions = {}

    for column in dataframe.columns:
        # Get column data
        col_data = dataframe[column]
        col_type = col_data.dtype
        sample_data = dataframe[column].dropna().sample(min(5, len(dataframe[column]))).tolist()
        sample_data_str = ", ".join(map(str, sample_data))

        # Prepare the prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""
                The following is a description of a dataset column:
                Column Name: {column}
                Data Type: {col_type}
                
                Please provide a detailed description of this column, including its potential meaning, use, and importance in a dataset. Use sample data to identify the column's meaning.
                
                Sample Data: {sample_data_str}
                
                Use the following format:

                For example:
                   "Product": "This column contains the name of the product. It is a text field and can be used for exact or partial matches.",
                   "Category": "This column contains the category of the product. It is a text field and can be used for exact or partial matches.",
                    "MedicalBenefits": "This column contains the medical benefits of the product. It is a text field and can be used for exact or partial matches.",
                    "CustomerRating": "This column contains the customer rating of the product. It is a numerical field and can be used for exact matches or range comparisons.",
                    "PurchaseFrequency": "This column contains the frequency of product purchase. It is a text field and can be used for exact or partial matches.",
                    "description": "This column contains the description of the product. It is a text field and can be used for exact or partial matches."
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

        chain = LLMChain(prompt=prompt, llm=llm)

        # Generate the description
        response = chain.run("Please provide a detailed description of each column in the given dataset.")

        # Extract the description from the response
        description = response.strip()

        # Add the description to the dictionary
        descriptions[column] = description

    # Return the dictionary of column descriptions
    return descriptions


#  TODO - Use the database Object for doing this
def store_descriptions_in_db(descriptions, numerical_columns, categorical_columns, db_driver: Union[SQLiteDriver, PostgresDriver]):
    conn = db_driver._db_connection
    if conn is None:
        raise ValueError("Database connection not established.")
    
    c = conn.cursor()

    # Create table for column descriptions
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS column_descriptions (
            column_name TEXT PRIMARY KEY,
            description TEXT
        )
    """
    )

    for column, description in descriptions.items():
        c.execute(
            """
            INSERT OR REPLACE INTO column_descriptions (column_name, description)
            VALUES (?, ?)
        """,
            (column, description),
        )

    # Create table for numerical and categorical columns
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS column_types (
            column_name TEXT PRIMARY KEY,
            column_type TEXT
        )
    """
    )

    for column in numerical_columns:
        c.execute(
            """
            INSERT OR REPLACE INTO column_types (column_name, column_type)
            VALUES (?, ?)
        """,
            (column, "numerical"),
        )

    for column in categorical_columns:
        c.execute(
            """
            INSERT OR REPLACE INTO column_types (column_name, column_type)
            VALUES (?, ?)
        """,
            (column, "categorical"),
        )

    conn.commit()
    conn.close()

def get_chroma_collection(
        collection_name: str, 
        db_driver: Union[SQLiteDriver, PostgresDriver], 
        dataset_table_name: str,
        categorical_columns: List[str],
        numerical_columns: List[str],
        primary_key: Optional[str]
    ) -> chromadb.Collection:
    """Gets the chroma collection.

    Returns:
        Chroma: Chroma collection.
    """
    chroma_client = chromadb.PersistentClient()
    collections = [col.name for col in chroma_client.list_collections()]

    if collection_name in collections:
        print(f"Collection '{collection_name}' already exists, getting existing collection...")
        chroma_collection = chroma_client.get_collection(collection_name)
    else:
        print(f"Collection '{collection_name}' does not exists, Creating new collection...")
        collection = chroma_client.create_collection(collection_name)

        data = db_driver.fetch_data_from_database(dataset_table_name)

        if data is None:
            raise ValueError("No data found in the database.")

        # TODO - Modify this project specific stuff to work with the data driver

        # TODO - Get column names from the database
        data["combined_text"] = data[categorical_columns].apply(lambda x: " ".join(x.dropna().astype(str)), axis=1)
        combined_text = data["combined_text"].tolist()
        data["meta_data"] = data[numerical_columns].apply(lambda x: " ".join(x.dropna().astype(str)), axis=1)
        metadata = data["meta_data"].tolist()

        if not primary_key:
            primary_key = data.columns[0]

        for text, pri_key, meta in zip(combined_text, data[primary_key], metadata):
            chroma_collection = collection.add(
                documents=text,
                ids=pri_key,
                metadatas={
                    f"product details: {str(numerical_columns)} ": meta
                },
            )

        chroma_collection = chroma_client.get_collection(collection_name)
    return chroma_collection

def generate_column_description(df: pd.DataFrame, db_driver: Union[SQLiteDriver, PostgresDriver]):

    # Get column descriptions
    column_descriptions = get_column_descriptions(
        dataframe=df
    )

    # Identify numerical and categorical columns
    numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

    # Store descriptions and column types in the database
    store_descriptions_in_db(
        descriptions=column_descriptions, numerical_columns=numerical_columns, categorical_columns=categorical_columns, db_driver=db_driver
    )
    
    print(column_descriptions)
    print("Column descriptions and column types stored in the database.")
