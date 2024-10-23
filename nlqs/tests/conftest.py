import sqlite3
from pathlib import Path
from typing import Callable, List
from unittest.mock import Mock, patch

from langchain_openai import OpenAIEmbeddings
import pandas as pd
import psycopg2
from pydantic import SecretStr
import pytest

from nlqs.database.postgres import PostgresConnectionConfig, PostgresDriver
from nlqs.database.sqlite import SQLiteConnectionConfig, SQLiteDriver
from nlqs.parameters import OPENAI_API_KEY
from nlqs.vectordb_driver import ChromaDBConfig, VectorDBDriver


@pytest.fixture
def sqlite_driver():
    sqlite_config = SQLiteConnectionConfig(db_file=Path("aegion.db"), dataset_table_name="new_dataset")

    db_driver = SQLiteDriver(sqlite_config=sqlite_config)

    return db_driver


@pytest.fixture(scope="function")
def setup_database():
    """Setup method to create a test database and driver instance."""
    test_db_file = Path("test_database.db")
    sqlite_config = SQLiteConnectionConfig(db_file=test_db_file, dataset_table_name="test_table")
    driver = SQLiteDriver(sqlite_config)

    # Create a test database and table
    with sqlite3.connect(test_db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY,
                name TEXT,
                value REAL
            )
            """
        )
        cursor.execute("INSERT INTO test_table (name, value) VALUES ('John', 10.5)")
        cursor.execute("INSERT INTO test_table (name, value) VALUES ('Jane', 20.0)")

        # Create a table for column descriptions
        cursor.execute(
            """
            CREATE TABLE column_descriptions (
                column_name TEXT PRIMARY KEY,
                description TEXT
            )
            """
        )
        cursor.execute(
            "INSERT INTO column_descriptions (column_name, description) VALUES (?, ?)",
            ("id", "Unique identifier"),
        )
        cursor.execute(
            "INSERT INTO column_descriptions (column_name, description) VALUES (?, ?)",
            ("name", "Name"),
        )
        cursor.execute(
            "INSERT INTO column_descriptions (column_name, description) VALUES (?, ?)",
            ("value", "Value"),
        )

        # Create a table for column types
        cursor.execute(
            """
            CREATE TABLE column_types (
                column_name TEXT PRIMARY KEY,
                column_type TEXT
            )
            """
        )
        cursor.execute(
            "INSERT INTO column_types (column_name, column_type) VALUES (?, ?)",
            ("id", "numerical"),
        )
        cursor.execute(
            "INSERT INTO column_types (column_name, column_type) VALUES (?, ?)",
            ("name", "categorical"),
        )
        cursor.execute(
            "INSERT INTO column_types (column_name, column_type) VALUES (?, ?)",
            ("value", "numerical"),
        )

        # Create a table with multiple primary keys
        cursor.execute(
            """
            CREATE TABLE test_table_multiple_pk (
                id1 INTEGER,
                id2 INTEGER,
                name TEXT,
                PRIMARY KEY (id1, id2)
            )
            """
        )

    yield driver

    # Cleanup method to remove the test database file
    test_db_file.unlink(missing_ok=True)


@pytest.fixture
def pg_config():
    return PostgresConnectionConfig(
        host="localhost",
        port=5432,
        user="postgres",
        password="password",
        database_name="test_db",
        dataset_table_name="test_table",
        uri_column="url",
    )


@pytest.fixture
def driver(pg_config):
    return PostgresDriver(pg_config)


@pytest.fixture
def mock_connection():
    mock_connection = Mock(spec=psycopg2.extensions.connection)
    mock_cursor = Mock(spec=psycopg2.extensions.cursor)
    mock_connection.cursor.return_value = mock_cursor
    return mock_connection


@pytest.fixture(autouse=True)
def patch_psycopg2_connect(mock_connection):
    with patch("psycopg2.connect", return_value=mock_connection) as mock:
        yield mock


@pytest.fixture(scope="session")
def chroma_config():
    return ChromaDBConfig(persist_path=Path("./chroma"), is_local=True)


@pytest.fixture(scope="session")
def vectordb_driver(chroma_config, embedding_function):

    VectorDBDriver.purge_nlqs_vectordb(chroma_config)

    VectorDBDriver.initialize_nlqs_vectordb(chroma_config)

    # Load the column and data description datasets
    column_info_df = pd.read_csv("./nlqs/tests/data/column_descriptions_with_embeddings.tsv", sep="\t")
    data_info_df = pd.read_csv("./nlqs/tests/data/data_descriptions_with_embeddings.tsv", sep="\t")
    table_info_df = pd.read_csv("./nlqs/tests/data/table_descriptions_with_embeddings.tsv", sep="\t")

    VectorDBDriver.populate_nlqs_column_info(
        chroma_config,
        column_info_df,
    )

    VectorDBDriver.populate_nlqs_dataset_info(
        chroma_config,
        data_info_df,
    )

    VectorDBDriver.populate_nlqs_table_info(
        chroma_config,
        table_info_df,
    )

    vectordb_driver = VectorDBDriver(chroma_config, embedding_function)

    return vectordb_driver


@pytest.fixture(scope="session")
def embedding_function() -> (
    Callable[[str], List[float]]
):  # -> Callable[..., List[float]]:# -> Callable[..., List[float]]:

    # Initialize the Embedding model
    # TODO: Rearchitect this so that we can switch models
    embedding_model = OpenAIEmbeddings(api_key=SecretStr(OPENAI_API_KEY), model="text-embedding-ada-002")

    # Create an embedding function
    embedding_function = embedding_model.embed_query

    return embedding_function
