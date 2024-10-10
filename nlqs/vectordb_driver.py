"""
VectorDB Data Schema

Column Info Collection (name: nlqs_column_info)
{
    "document": "description : This column contains the description of the product. It is a text field and can be used for exact or partial matches.",
    "embedding": [0.1, 0.2, 0.3, 0.4, ... , 0.5],
    "metadata": {
        "db_name": "Location of the original database",
        "table_name": "Location of the original table",
        "column_name": "Column name",
        "column_type": "descriptive"
    }
}

Dataset Collection (name: nlqs_descriptive_data)
{
    "document": "Raw data from the dataset",
    "embedding": [0.1, 0.2, 0.3, 0.4, ... , 0.5],
    "metadata": {
        "db_name": "Location of the original database",
        "table_name": "Location of the original table",
        "lookup_key_column_name": "Primary key column name",
        "lookup_key_column_value": "Primary key column value
    }
}

Table Descriptions Collection (name: nlqs_table_descriptions)
{
    "document": "Description of the table",
    "embedding": [0.1, 0.2, 0.3, 0.4, ... , 0.5],
    "metadata": {
        "db_name": "Location of the original database",
        "table_name": "Location of the original table",
    }
}
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, List, Mapping, Optional, Tuple, TypedDict
import chromadb
from chromadb.config import Settings
from pandas import DataFrame
from tqdm import tqdm


DEFAULT_COLUMN_INFO_COLLECTION_NAME = "nlqs_column_info"
DEFAULT_DATASET_COLLECTION_NAME = "nlqs_descriptive_data"
DEFAULT_TABLE_DESCRIPTION_COLLECTION_NAME = "nlqs_table_descriptions"

DEFAULT_BATCH_SIZE = 10

class ColumnType(Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    DESCRIPTIVE = "descriptive"
    IDENTIFIER = "identifier"


class DataCollectionMetadata(TypedDict):
    db_name: str
    table_name: str
    lookup_key_column_name: str
    lookup_key_column_value: str


class ColumnInfoMetadata(TypedDict):
    db_name: str
    table_name: str
    column_name: str
    column_type: ColumnType


class TableDescriptionMetadata(TypedDict):
    db_name: str
    table_name: str


@dataclass
class ChromaDBConfig:
    table_description_collection_name: str = DEFAULT_TABLE_DESCRIPTION_COLLECTION_NAME
    column_info_collection_name: str = DEFAULT_COLUMN_INFO_COLLECTION_NAME
    dataset_collection_name: str = DEFAULT_DATASET_COLLECTION_NAME
    persist_path: Path = Path("./chroma")
    host: str = "localhost"
    port: int = 8000
    is_local: bool = True
    username: Optional[str] = None
    password: Optional[str] = None


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
            self.chroma_client = chromadb.HttpClient(
                port=chroma_config.port, 
                host=chroma_config.host, 
                settings=Settings(
                chroma_client_auth_provider="chromadb.auth.basic.BasicAuthClientProvider",
                chroma_client_auth_credentials=f"{chroma_config.username}:{chroma_config.password}",
            ),)


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

    @property
    def column_info_collection(self) -> chromadb.Collection:
        """ Get the column info collection.

        Returns:
            chromadb.Collection: Column info collection
        """

        collection = self.get_chroma_collection(self.chroma_config.column_info_collection_name)
        if collection is None:
            raise ValueError(f"Error: Collection '{self.chroma_config.column_info_collection_name}' does not exist.")
        return collection
    
    @property
    def dataset_collection(self) -> chromadb.Collection:
        """ Get the dataset collection.

        Returns:
            chromadb.Collection: Dataset collection
        """

        collection = self.get_chroma_collection(self.chroma_config.dataset_collection_name)
        if collection is None:
            raise ValueError(f"Error: Collection '{self.chroma_config.dataset_collection_name}' does not exist.")
        return collection


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

        # Step 1: Lookup and get the closest column name from the collection using a 
        # combination of the user's description and sample data strings
        column_info_collection = self.column_info_collection
        if not column_info_collection:
            raise ValueError("Column info collection does not exist.")

        # Create a description package
        description_package = f"""
        Closest Column Name: {approximate_column_name}
        User's Description: {users_description}
        Sample Data: {', '.join(sample_data_strings)}
        """

        # TODO: Figure out which embedding to use

        # Use the formatted string to find the closest column
        results = column_info_collection.query(query_texts=[description_package], n_results=1)

        if not results:
            raise ValueError(f"No matching column found for {approximate_column_name}.")

        metadatas: List[List[Mapping[str, str | int | float | bool]]] | None = results['metadatas']
        if not metadatas:
            raise ValueError(f"No metadata found in chromadb query result for {approximate_column_name}.")
        
        # TODO: Figure out if this is the correct way to get the closest column name
        closest_column_name = metadatas[0][0]["column_name"]

        if type(closest_column_name) is not str:
            raise ValueError(f"Closest column name is not a string for {approximate_column_name}. Extracted Info: {closest_column_name}.")

        return closest_column_name, self.get_column_type(closest_column_name)
    

    def get_column_type(self, column_name: str) -> ColumnType:
        """ Get the column type for the given column name.

        Args:
            column_name (str): Column name

        Returns:
            ColumnType: Column type
        """

        raise NotImplementedError("This method is not implemented yet.")


    def store_column_info_in_db(
        self,
        column_name: str,
        description: str,
        column_type: ColumnType,
    ) -> None:
        """ Store the column information in the database.

        Args:
            column_name (str): Column name
            description (str): Column description
            column_type (ColumnType): Column type
        """

        raise NotImplementedError("This method is not implemented yet.")


    @staticmethod
    def initialize_nlqs_vectordb(
        chroma_config: ChromaDBConfig,
    ) -> None:
        """ Initialize the NLQS VectorDB collections.

        Args:
            chroma_config (ChromaDBConfig): ChromaDB configuration
            column_info_collection_name (str): Column info collection name
            dataset_collection_name (str): Dataset collection name
        """

        # Create a new driver instance
        driver = VectorDBDriver(chroma_config)

        # Create the NLQS collections
        driver.chroma_client.create_collection(chroma_config.column_info_collection_name)
        driver.chroma_client.create_collection(chroma_config.dataset_collection_name)

    
    @staticmethod
    def populate_nlqs_vectordb(
        chroma_config: ChromaDBConfig,
        column_info: Optional[DataFrame] = None,
        dataset_info: Optional[DataFrame] = None,
        table_info: Optional[DataFrame] = None,
        batch_size:int = DEFAULT_BATCH_SIZE,
        
    ) -> None:
        """ Populate the NLQS VectorDB collections

        Args:
            chroma_config (ChromaDBConfig): ChromaDB configuration
            column_info (DataFrame): Column information
            dataset_info (DataFrame): Dataset information
        """

        vectordb_driver = VectorDBDriver(chroma_config)

        if column_info is not None:

            # Ids of the rows
            ids = [str(i+1) for i in range(len(column_info))]

            for index in tqdm(range(0, len(column_info), batch_size)):


                descriptions = column_info["description"].astype(str).tolist()[index:index+batch_size]
                db_names = column_info["db_name"].astype(str).tolist()[index:index+batch_size]
                table_names = column_info["table_name"].astype(str).tolist()[index:index+batch_size]
                column_names = column_info["column_name"].astype(str).tolist()[index:index+batch_size]
                column_types = column_info["column_type"].astype(str).tolist()[index:index+batch_size]
                embeddings = column_info["embedding"].tolist()[index:index+batch_size]

                # Create metadata objects
                metadatas = []
                for i in range(len(descriptions)):
                    metadatas.append({
                        "db_name": db_names[i],
                        "table_name": table_names[i],
                        "column_name": column_names[i],
                        "column_type": column_types[i]
                    })
                
                # Populate the column info collection
                vectordb_driver.column_info_collection.add(
                    ids=ids[index:index+batch_size],
                    documents=descriptions,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )

        
        if dataset_info is not None:
            # Populate the dataset collection
            ids = [str(i+1) for i in range(len(dataset_info))]

            for index in tqdm(range(0, len(dataset_info), batch_size)):

                documents = dataset_info["description"].astype(str).tolist()[index:index+batch_size]
                db_names = dataset_info["db_name"].astype(str).tolist()[index:index+batch_size]
                table_names = dataset_info["table_name"].astype(str).tolist()[index:index+batch_size]
                column_names = dataset_info["column_name"].astype(str).tolist()[index:index+batch_size]
                lookup_key_column_names = dataset_info["lookup_key_column_name"].astype(str).tolist()[index:index+batch_size]
                lookup_key_column_values = dataset_info["lookup_key_column_value"].astype(str).tolist()[index:index+batch_size]
                embeddings = dataset_info["embedding"].tolist()[index:index+batch_size]

                # Create metadata objects
                metadatas = []
                for i in range(len(documents)):
                    metadatas.append({
                        "db_name": db_names[i],
                        "table_name": table_names[i],
                        "column_name": column_names[i],
                        "lookup_key_column_name": lookup_key_column_names[i],
                        "lookup_key_column_value": lookup_key_column_values[i]
                    })
                
                # Populate the dataset collection
                vectordb_driver.dataset_collection.add(
                    ids=ids[index:index+batch_size],
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )


        if table_info is not None:
            # Populate the table description collection
            ids = [str(i+1) for i in range(len(table_info))]

            for index in tqdm(range(0, len(table_info), batch_size)):
                documents = table_info["description"].astype(str).tolist()[index:index+batch_size]
                db_names = table_info["db_name"].astype(str).tolist()[index:index+batch_size]
                table_names = table_info["table_name"].astype(str).tolist()[index:index+batch_size]
                embeddings = table_info["embedding"].tolist()[index:index+batch_size]

                # Create metadata objects
                metadatas = []
                for i in range(len(documents)):
                    metadatas.append({
                        "db_name": db_names[i],
                        "table_name": table_names[i]
                    })
                
                # Populate the table description collection
                vectordb_driver.get_chroma_collection(chroma_config.table_description_collection_name).add(
                    ids=ids[index:index+batch_size],
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )


        raise NotImplementedError("This method is not implemented yet.")


