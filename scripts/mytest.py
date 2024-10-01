from pathlib import Path
from typing import Dict, Union
import chromadb
from scripts.parameters import CHROMA_COLLECTION_NAME, OUTPUT_COLUMNS, SQL_TABLE_NAME, SQLITE_DB_FILE
from nlqs.database.postgres import PostgresConnectionConfig, PostgresDriver
from nlqs.database.sqlite import SQLiteConnectionConfig, SQLiteDriver
from nlqs.nlqs import ChromaDBConfig


def columns_chroma_lookup(
    db_driver: Union[SQLiteDriver, PostgresDriver],
    chroma_client,
    categorical_data: Dict[str, str],
    descriptive_data: Dict[str, str],
):
    """
    This function checks if the given categorical and descriptive data exists in the chroma collection.
    """
    collection = chroma_client.get_collection(name="column_info")

    # Check for categorical data
    for column_name, data_value in categorical_data.items():
        categorical_columns = db_driver.execute_query(
            "SELECT column_name FROM column_metadata WHERE column_type = 'categorical'"
        )
        if not categorical_columns:
            raise ValueError("No categorical columns found in the database.")
        categorical_columns = [row[0] for row in categorical_columns]
        if column_name not in categorical_columns:
            return False

        results = collection.query(query_texts=[data_value], n_results=1)
        if results["ids"][0]:
            retrieved_column = results.get("ids")[0][0]
            print(f"Original column data: {data_value}")
            print(f"LLM column: {column_name}")
            print(f"Chroma column: {retrieved_column}")

            if column_name != retrieved_column:
                return False
        else:
            return False

    # Check for descriptive data
    for column_name, data_value in descriptive_data.items():
        descriptive_columns = db_driver.execute_query(
            "SELECT column_name FROM column_metadata WHERE column_type = 'descriptive'"
        )
        if not descriptive_columns:
            raise ValueError("No descriptive columns found in the database.")
        descriptive_columns = [row[0] for row in descriptive_columns]
        if column_name not in descriptive_columns:
            return False

        results = collection.query(query_texts=[data_value], n_results=1)
        if results["ids"][0]:
            retrieved_column = results.get("ids")[0][0]
            print(f"Original column data: {data_value}")
            print(f"LLM column: {column_name}")
            print(f"Chroma column: {retrieved_column}")

            if column_name != retrieved_column:
                return False
        else:
            return False

    return True


if __name__ == "__main__":

    sqlite_config = SQLiteConnectionConfig(
        db_file=Path(SQLITE_DB_FILE), dataset_table_name=SQL_TABLE_NAME, uri_column="URL", output_columns=OUTPUT_COLUMNS
    )

    # postgres_config = PostgresConnectionConfig(
    #     host="aws-0-us-east-1.pooler.supabase.com",
    #     port=6543,
    #     user="postgres.xdvwtpqclkedpktjsrzc",
    #     password="aOoDlcdghQ39Gkjr",
    #     database_name="postgres",
    #     dataset_table_name="new_dataset",
    #     uri_column="URL",
    # )

    connection_config = sqlite_config

    if isinstance(connection_config, SQLiteConnectionConfig):
        connection_driver = SQLiteDriver(connection_config)
    elif isinstance(connection_config, PostgresConnectionConfig):
        connection_driver = PostgresDriver(connection_config)
    else:
        raise ValueError("Invalid connection configuration")

    connection_driver.connect()

    # ChromaDB configuration
    chroma_config = ChromaDBConfig(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_path=Path("../chroma"),
    )

    chroma_type = chroma_config.is_local

    if chroma_type:
        chroma_client = chromadb.PersistentClient(path=str(chroma_config.persist_path))
    else:
        chroma_client = chromadb.HttpClient(port=chroma_config.port, host=chroma_config.host)

    categorical_data = {"Category": "Gummies"}

    descriptive_data = {"MedicalBenefitsReported": "Pain relief", "Product": "Moroccan Mint"}

    test_result = columns_chroma_lookup(connection_driver, chroma_client, categorical_data, descriptive_data)
    print(f"Test Result: {test_result}")
