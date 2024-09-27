from typing import Dict, List, Optional, Tuple, Union
from nlqs.database.postgres import PostgresDriver
from nlqs.database.sqlite import SQLiteDriver
import chromadb
from nlqs.nlqs import ChromaDBConfig
from nlqs.database.postgres import PostgresConnectionConfig
from nlqs.database.sqlite import SQLiteConnectionConfig
from pathlib import Path
from scripts.parameters import (
    CHROMA_COLLECTION_NAME,
    SQL_TABLE_NAME,
    SQL_TABLE_NAME,
    SQLITE_DB_FILE,
    SUPABASE_DATABASE_NAME,
    SUPABASE_HOST,
    SUPABASE_PASSWORD,
    SUPABASE_PORT,
    SUPABASE_USER,
    URL_COLUMN,
    VECTORDB_HOST,
    VECTORDB_PORT,
)


def generate_chroma_collection(
    collection_name: str,
    client,
    db_driver: Union[SQLiteDriver, PostgresDriver],
    primary_key: Optional[str],
    descriptive_columns: List[str],
    categorical_columns: List[str],
) -> Tuple[chromadb.Collection, chromadb.Collection]:
    """Retrieves data from the chroma collection, if there is no chroma collection it creates one.

    Args:
        collection_name (str): name of chroma collection for descriptive data.
        client (_type_): chroma client
        db_driver (Union[SQLiteDriver, PostgresDriver]): database driver.
        primary_key (Optional[str]): primary key of the database.
        descriptive_columns (List[str]): descriptive columns in the database.
        categorical_columns (List[str]): categorical columns in the database.

    Returns:
        Tuple[chromadb.Collection, chromadb.Collection]: A tuple containing the chroma collection for descriptive data and the chroma collection for categorical data.
    """

    collections = [col.name for col in client.list_collections()]

    # Descriptive Data Collection
    if collection_name in collections:
        print(f"Collection '{collection_name}' already exists, getting existing collection...")
        descriptive_collection = client.get_collection(collection_name)
    else:
        print(f"Collection '{collection_name}' does not exists, Creating new collection...")
        descriptive_collection = client.create_collection(collection_name)

        data = db_driver.fetch_data_from_database(db_driver.db_config.dataset_table_name)

        if data is None:
            raise ValueError("No data found in the database.")

        if not primary_key:
            primary_key = data.columns[0]

        for index, row in data.iterrows():
            # Extract the primary key value
            pri_key = str(row[primary_key])

            for column in descriptive_columns:
                # Extract the text for the current column and row
                text = [str(row[column])]

                # Create the ID for the current column and row
                id = f"{column}_{pri_key}"

                print(f"Adding {id} to chroma")

                # Create the metadata dictionary
                meta = {
                    primary_key: pri_key,
                    "table_name": db_driver.db_config.dataset_table_name,
                    "column_name": column,
                }

                # Add the data to the Chroma collection
                descriptive_collection.add(
                    documents=text,
                    ids=id,
                    metadatas=meta,
                )

    # Categorical Data Collection
    categorical_collection_name = "categorical_data"  # Default name for categorical collection
    if categorical_collection_name in collections:
        print(f"Collection '{categorical_collection_name}' already exists, getting existing collection...")
        categorical_collection = client.get_collection(categorical_collection_name)
    else:
        print(f"Collection '{categorical_collection_name}' does not exists, Creating new collection...")
        categorical_collection = client.create_collection(categorical_collection_name)

        data = db_driver.fetch_data_from_database(db_driver.db_config.dataset_table_name)

        for column in categorical_columns:
            # Get unique values for the categorical column
            unique_values = data[column].unique().astype(str).tolist()

            for value in unique_values:
                # Create the ID for the current column and value
                id = f"{column}_{value}"

                print(f"Adding {id} to chroma")

                # Create the metadata dictionary
                meta = {
                    "table_name": db_driver.db_config.dataset_table_name,
                    "column_name": column,
                }

                # Add the data to the Chroma collection
                categorical_collection.add(
                    documents=[value],
                    ids=id,
                    metadatas=meta,
                )

    return descriptive_collection, categorical_collection


if __name__ == "__main__":

    # SQLite configuration
    # connection_config = SQLiteConnectionConfig(db_file=Path(SQLITE_DB_FILE), dataset_table_name=SQL_TABLE_NAME)

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
    # chroma_config = ChromaDBConfig(
    #     collection_name=CHROMA_COLLECTION_NAME,
    #     persist_path=Path("D://work//conversational-ai-modules-monorepo//chroma"),
    # )  # local chroma

    # remote config
    chroma_config = ChromaDBConfig(
        collection_name=CHROMA_COLLECTION_NAME, is_local=False, host=VECTORDB_HOST, port=int(VECTORDB_PORT)
    )

    chroma_type = chroma_config.is_local
    print(f"chroma_config.persist_path: {chroma_config.persist_path.parent.absolute()}")
    if chroma_type:
        chroma_client = chromadb.PersistentClient(path=str(chroma_config.persist_path))
    else:
        chroma_client = chromadb.HttpClient(port=chroma_config.port, host=chroma_config.host)

    column_descriptions, numerical_columns, categorical_columns, descriptive_columns = (
        connection_driver.retrieve_descriptions_and_types_from_db()
    )

    if column_descriptions == {}:
        raise ValueError("No data found in the database. Generate Column descriptions.")

    primary_key = connection_driver.get_primary_key(connection_driver.db_config.dataset_table_name)

    generate_chroma_collection(
        collection_name=chroma_config.collection_name,
        client=chroma_client,
        db_driver=connection_driver,
        primary_key=primary_key,
        descriptive_columns=descriptive_columns,
        categorical_columns=categorical_columns,
    )
