import pytest
from pathlib import Path
from unittest.mock import patch, Mock
import pandas as pd
import sqlite3
from sqlalchemy.exc import SQLAlchemyError
from nlqs.database.sqlite import SQLiteConnectionConfig, SQLiteDriver
from sqlalchemy import text


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


def test_connect_successful(setup_database):
    """Test successful connection to the database."""
    driver = setup_database
    driver.connect()
    assert driver.engine is not None
    assert driver.Session is not None


@patch("nlqs.database.sqlite.create_engine")  # Patch where 'create_engine' is called
def test_connect_failed(mock_create_engine, setup_database):
    """Test connection failure."""
    driver = setup_database
    mock_create_engine.side_effect = SQLAlchemyError("Error")
    with pytest.raises(SQLAlchemyError):
        driver.connect()


def test_disconnect_successful(setup_database):
    """Test successful disconnection from the database."""
    driver = setup_database
    driver.connect()
    driver.disconnect()
    # No need for assertions, SQLAlchemy handles disconnection internally


def test_execute_query_successful(setup_database):
    """Test successful execution of a SQL query."""
    driver = setup_database
    driver.connect()
    result = driver.execute_query("SELECT name FROM test_table WHERE value > 15")
    assert result[0][0] == "Jane"


def test_execute_query_with_error(setup_database):
    """Test handling of errors during query execution."""
    driver = setup_database
    driver.connect()
    with pytest.raises(SQLAlchemyError):
        driver.execute_query("SELECT * FROM nonexistent_table")


def test_retrieve_descriptions_and_types_from_db_successful(setup_database):
    """Test successful retrieval of descriptions and types from the database."""
    driver = setup_database
    driver.connect()

    expected_descriptions = {"id": "Unique identifier", "name": "Name", "value": "Value"}
    expected_numerical_columns = ["id", "value"]
    expected_categorical_columns = ["name"]

    descriptions, numerical_columns, categorical_columns = driver.retrieve_descriptions_and_types_from_db()

    assert descriptions == expected_descriptions
    assert numerical_columns == expected_numerical_columns
    assert categorical_columns == expected_categorical_columns


def test_retrieve_descriptions_and_types_from_db_with_error(setup_database):
    """Test handling of errors during descriptions and types retrieval."""
    driver = setup_database
    driver.connect()

    descriptions = {}
    numerical_columns = []
    categorical_columns = []

    # Drop the tables to simulate an error condition
    with driver.engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS column_descriptions"))
        conn.execute(text("DROP TABLE IF EXISTS column_types"))

    # Assert that an error is raised and handled gracefully
    with pytest.raises(SQLAlchemyError, match="no such table: column_descriptions"):
        descriptions, numerical_columns, categorical_columns = driver.retrieve_descriptions_and_types_from_db()

    # Assert that empty values are returned
    assert descriptions == {}
    assert numerical_columns == []
    assert categorical_columns == []


def test_get_database_columns(setup_database):
    """Test retrieval of database columns in order."""
    driver = setup_database
    driver.connect()
    columns = driver.get_database_columns("test_table")
    assert columns == ["id", "name", "value"]


def test_validate_query_valid_query(setup_database):
    """Test validation of a valid SQL query."""
    driver = setup_database
    driver.connect()
    is_valid = driver.validate_query("SELECT name, value FROM test_table WHERE id = 1")
    assert is_valid


def test_validate_query_invalid_query(setup_database):
    """Test validation of an invalid SQL query."""
    driver = setup_database
    driver.connect()
    is_valid = driver.validate_query("SELECT * FROM nonexistent_table")
    assert not is_valid


def test_check_table_exists_existing_table(setup_database):
    """Test checking for an existing table."""
    driver = setup_database
    driver.connect()
    assert driver.check_table_exists("test_table")


def test_check_table_exists_nonexistent_table(setup_database):
    """Test checking for a nonexistent table."""
    driver = setup_database
    driver.connect()
    assert not driver.check_table_exists("nonexistent_table")


def test_fetch_data_from_database_successful(setup_database):
    """Test successful data fetching from the database."""
    driver = setup_database
    driver.connect()
    df = driver.fetch_data_from_database("test_table")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2  # Check if two rows were fetched


def test_fetch_data_from_database_nonexistent_table(setup_database):
    """Test data fetching from a nonexistent table."""
    driver = setup_database
    driver.connect()
    with pytest.raises(ValueError, match="Table 'nonexistent_table' does not exist in the database."):
        driver.fetch_data_from_database("nonexistent_table")


def test_get_primary_key(setup_database):
    """Test getting the primary key of a table."""
    driver = setup_database
    driver.connect()
    primary_key = driver.get_primary_key("test_table")
    assert primary_key == "id"


def test_get_primary_key_no_primary_key(setup_database):
    """Test getting the primary key when the table has no primary key."""
    driver = setup_database
    driver.connect()
    with pytest.raises(ValueError, match="No primary key found"):
        driver.get_primary_key("test_table_no_pk")


def test_get_primary_key_multiple_primary_keys(setup_database):
    """Test getting the primary key when the table has multiple primary keys."""
    driver = setup_database
    driver.connect()
    with pytest.raises(ValueError, match="Multiple primary keys found"):
        driver.get_primary_key("test_table_multiple_pk")
