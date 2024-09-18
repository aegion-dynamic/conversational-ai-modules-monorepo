import pytest
from unittest.mock import patch, Mock
import pandas as pd
import psycopg2
from nlqs.database.postgres import PostgresConnectionConfig, PostgresDriver


@pytest.fixture
def pg_config():
    return PostgresConnectionConfig(
        host="localhost",
        port=5432,
        user="postgres",
        password="password",
        database_name="test_db",
        dataset_table_name="test_table",
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


def test_connect_successful(driver, mock_connection):
    """Test successful connection to the database."""
    driver.connect()
    assert driver.db_connection == mock_connection
    assert driver.cursor == mock_connection.cursor.return_value


def test_connect_with_error(driver, mock_connection):
    """Test connection failure with psycopg2.Error."""
    mock_connection.cursor.side_effect = psycopg2.Error("Test error")
    with pytest.raises(psycopg2.Error):
        driver.connect()


def test_disconnect_successful(driver, mock_connection):
    """Test successful disconnection from the database."""
    driver.connect()
    driver.disconnect()
    mock_connection.close.assert_called_once()


def test_execute_query_successful(driver, mock_connection):
    """Test successful execution of a SQL query."""
    driver.connect()
    mock_connection.cursor.return_value.fetchall.return_value = [("Jane",), ("John",)]
    result = driver.execute_query("SELECT name FROM test_table WHERE value > 15")
    assert result == ["Jane", "John"]


def test_execute_query_with_error(driver, mock_connection):
    """Test handling of errors during query execution."""
    driver.connect()
    mock_connection.cursor.return_value.execute.side_effect = psycopg2.Error("Test error")
    with pytest.raises(psycopg2.Error):
        driver.execute_query("SELECT * FROM nonexistent_table")


def test_retrieve_descriptions_and_types_from_db_successful(driver, mock_connection):
    """Test successful retrieval of descriptions and types from the database."""
    driver.connect()
    mock_connection.cursor.return_value.fetchall.side_effect = [
        [
            ("id", "Unique identifier"),
            ("name", "Name of the person"),
            ("value", "Some value"),
        ],
        [("id", "numerical"), ("name", "categorical"), ("value", "numerical")],
    ]

    expected_descriptions = {
        "id": "Unique identifier",
        "name": "Name of the person",
        "value": "Some value",
    }
    expected_numerical_columns = ["id", "value"]
    expected_categorical_columns = ["name"]

    descriptions, numerical_columns, categorical_columns = driver.retrieve_descriptions_and_types_from_db()

    assert descriptions == expected_descriptions
    assert numerical_columns == expected_numerical_columns
    assert categorical_columns == expected_categorical_columns


def test_retrieve_descriptions_and_types_from_db_with_error(driver, mock_connection):
    """Test handling of errors during descriptions and types retrieval."""
    driver.connect()
    mock_connection.cursor.return_value.execute.side_effect = psycopg2.Error("Test error")

    descriptions, numerical_columns, categorical_columns = driver.retrieve_descriptions_and_types_from_db()

    assert descriptions == {}
    assert numerical_columns == []
    assert categorical_columns == []


def test_get_database_columns(driver, mock_connection):
    """Test retrieval of database columns in order."""
    driver.connect()
    mock_connection.cursor.return_value.fetchall.return_value = [
        (0, "id", "INTEGER", 1, None, 1),
        (1, "name", "TEXT", 1, None, 0),
        (2, "value", "REAL", 1, None, 0),
    ]
    columns = driver.get_database_columns("test_table")
    assert columns == ["id", "name", "value"]


def test_validate_query_valid_query(driver, mock_connection):
    """Test validation of a valid SQL query."""
    driver.connect()
    mock_connection.cursor.return_value.fetchone.return_value = True
    mock_connection.cursor.return_value.fetchall.return_value = [("id",), ("name",), ("value",)]
    is_valid = driver.validate_query("SELECT name, value FROM test_table WHERE id = 1")
    assert is_valid is True


def test_validate_query_invalid_query(driver, mock_connection):
    """Test validation of an invalid SQL query for a non-existent table."""
    driver.connect()

    # Simulate no results for the table (table does not exist).
    mock_connection.cursor.return_value.fetchone.return_value = None  # No table found
    mock_connection.cursor.return_value.fetchall.return_value = []  # No columns found

    # Call the validate_query function with an invalid table
    is_valid = driver.validate_query("SELECT * FROM nonexistent_table")

    # Assert that the query is invalid since the table doesn't exist
    assert is_valid is False


def test_fetch_data_from_database_successful(driver, mock_connection):
    """Test successful data fetching from the database."""
    driver.connect()
    mock_df = pd.DataFrame({"id": [1, 2], "name": ["John", "Jane"], "value": [10.5, 20.0]})
    with patch("pandas.read_sql_query", return_value=mock_df) as mock_read_sql:
        df = driver.fetch_data_from_database("test_table")
    assert isinstance(df, pd.DataFrame)
    assert df.equals(mock_df)


def test_fetch_data_from_database_with_error(driver, mock_connection):
    """Test data fetching from the database with psycopg2.Error."""
    driver.connect()
    with patch("pandas.read_sql_query", side_effect=psycopg2.Error("Test error")) as mock_read_sql:
        df = driver.fetch_data_from_database("test_table")
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_get_primary_key(driver, mock_connection):
    """Test getting the primary key of a table."""
    driver.connect()
    mock_connection.cursor.return_value.fetchall.return_value = [(0, "id", "INTEGER", 1, None, 1)]
    primary_key = driver.get_primary_key("test_table")
    assert primary_key == "id"


def test_get_primary_key_no_primary_key(driver, mock_connection):
    """Test getting the primary key when the table has no primary key."""
    driver.connect()
    mock_connection.cursor.return_value.fetchall.return_value = [(0, "id", "INTEGER", 1, None, 0)]
    with pytest.raises(ValueError) as context:
        driver.get_primary_key("test_table")
    assert "No primary key found" in str(context.value)
