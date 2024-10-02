

from dataclasses import dataclass
from enum import Enum
from logging.config import IDENTIFIER
from pathlib import Path
from typing import List, Optional, Tuple, Union
from unittest.mock import DEFAULT
import chromadb


DEFAULT_COLUMN_INFO_COLLECTION_NAME = "nlqs_column_info"
DEFAULT_DATASET_COLLECTION_NAME = "nlqs_dataset"

class ColumnType(Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    DESCRIPTIVE = "descriptive"
    IDENTIFIER = "identifier"


@dataclass
class ChromaDBConfig:
    collection_name: str
    persist_path: Path = Path("./chroma")
    host: str = "localhost"
    port: int = 8000
    is_local: bool = True


class VectorDBDriver:

    def __init__(self, chroma_config: ChromaDBConfig):
        """ Constructor for the VectorDBDriver

        Args:
            chroma_config (ChromaDBConfig): ChromaDB configuration
        """
        self.chroma_config = chroma_config

        chroma_type = chroma_config.is_local
        if chroma_type:
            self.chroma_client = chromadb.PersistentClient(path=str(chroma_config.persist_path))
        else:
            self.chroma_client = chromadb.HttpClient(port=chroma_config.port, host=chroma_config.host)


    def check_nlqs_collections_exists(
        self,
        column_info_collection_name: str = DEFAULT_COLUMN_INFO_COLLECTION_NAME,
        dataset_collection_name: str = DEFAULT_DATASET_COLLECTION_NAME,
    ) -> bool:
        """ Check if the NLQS collections exist, and return them if they do.

        Args:
            custom_column_data_collection_name (str): Custom column data collection name
            custom_dataset_collection_name (str): Custom dataset collection name

        Returns:
            Tuple[Optional[chromadb.Collection], Optional[chromadb.Collection]]: Tuple of custom column data collection and custom dataset collection
        """
        

        custom_column_data_collection = self.get_chroma_collection(column_info_collection_name)
        custom_dataset_collection = self.get_chroma_collection(dataset_collection_name)

        return (custom_column_data_collection is not None) and (custom_dataset_collection is not None)


    def get_chroma_collection(
        self,
        collection_name: str,
    ) -> Optional[chromadb.Collection]:
        """ Return the Chroma collection if it exists, otherwise raises an error.

        Args:
            collection_name (str): Collection name

        Raises:
            ValueError: If the collection does not exist

        Returns:
            chromadb.Collection: Chroma collection
        """

        try:
            chroma_collection = self.chroma_client.get_collection(collection_name)
            print(f"Collection '{collection_name}' already exists, getting existing collection...")

        except ValueError as e:
            print(f"Collection '{collection_name}' does not exists")

            return None
        
        return chroma_collection


    def get_closest_column_from_description(
            self, 
            approximate_column_name: str, 
            users_description: str, 
            sample_data_strings: List[str]
        ) -> Tuple[str, ColumnType]:
        """ Get the closest column name from the description provided by the user.

        Args:
            approximate_column_name (str): Approximate column name
            users_description (str): User's description
            sample_data_strings (List[str]): Sample data strings

        Returns:
            str: Closest column name
        """

        # Step 1: Lookup and get the closest column name from the collection using a 
        # combination of the user's description and sample data strings
        # Step 2: Replace the approximate column name with the closest column name



        raise NotImplementedError("This method is not implemented yet.")
    
    def get_column_type(self, column_name: str) -> ColumnType:
        """ Get the column type for the given column name.

        Args:
            column_name (str): Column name

        Returns:
            ColumnType: Column type
        """

        raise NotImplementedError("This method is not implemented yet.")

    # collection = client.create_collection(collection_name)

    # data = db_driver.fetch_data_from_database(db_driver.db_config.dataset_table_name)

    # categorical_columns = data.select_dtypes(include=["object"]).columns.tolist()

    # if data is None:
    #     raise ValueError("No data found in the database.")

    # if not primary_key:
    #     primary_key = data.columns[0]

    # for index, row in data.iterrows():
    #     # Extract the primary key value
    #     pri_key = str(row[primary_key])

    #     for column in categorical_columns:
    #         # Extract the text for the current column and row
    #         text = [str(row[column])]

    #         # Create the ID for the current column and row
    #         id = f"{column}_{pri_key}"

    #         print(f"id: {id}")

    #         # Create the metadata dictionary
    #         meta = {
    #             "id": pri_key,
    #             "table_name": db_driver.db_config.dataset_table_name,
    #             "column_name": column,
    #         }

    #         # Add the data to the Chroma collection
    #         chroma_collection = collection.add(
    #             documents=text,
    #             ids=id,
    #             metadatas=meta,
    #         )

    # chroma_collection = client.get_collection(collection_name)
