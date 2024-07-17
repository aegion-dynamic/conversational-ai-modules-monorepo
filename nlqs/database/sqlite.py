import logging
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from discord_bot.parameters import LOGGER_FILE, SQLITE_DB_FILE
from nlqs.database.driver import AbstractDriver

# Create a logger object
logger = logging.getLogger(__name__)

# Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR)
logger.setLevel(logging.INFO)

# Create a file handler to save logs
file_handler = logging.FileHandler(LOGGER_FILE)

# Create a formatter to format the log messages
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Add the formatter to the file handler
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)


class SQLiteDriver(AbstractDriver):
    def __init__(self, config: Dict):
        self.db_file = config.get("db_file", SQLITE_DB_FILE)
        self.conn = None
        self.cursor = None

    def connect(self):
        try:
            self.conn = sqlite3.connect(self.db_file)
            self.cursor = self.conn.cursor()
            logger.info("Connected to SQLite database.")
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database: {e}")
            raise

    def disconnect(self):
        if self.conn:
            self.conn.close()
            logger.info("Disconnected from SQLite database.")

    def execute_query(self, query: str) -> str:
        """Executes the SQL query and returns the result.

        Args:
            query (str): the SQL query.

        Returns:
            str: the result of the query.
        """
        try:
            self.cursor.execute(query)
            result = self.cursor.fetchall()
            self.conn.commit()
            result_str = str(result)
            logger.info(f"Query executed successfully: {result_str}")
            return result_str if result else "No results found."
        except sqlite3.Error as e:
            error_message = f"Error executing SQL query: {e}"
            logger.error(error_message)
            return error_message

    def retrieve_descriptions_and_types_from_db(self) -> Tuple[Dict[str, str], List[str], List[str]]:
        """Retrieves descriptions and types from the SQLite database.

        Args:
            db_file (SQLite database, optional): SQLite database to store all the tables. Defaults to SQLITE_DB_FILE.

        Returns:
            Tuple[List[str], List[str], List[str]]: Return descriptions, numerical_columns, categorial_columns
        """
        # Retrieve descriptions
        self.cursor.execute("SELECT column_name, description FROM column_descriptions")
        description_rows = self.cursor.fetchall()
        descriptions = {row[0]: row[1] for row in description_rows}

        # Retrieve column types
        self.cursor.execute("SELECT column_name, column_type FROM column_types")
        type_rows = self.cursor.fetchall()
        numerical_columns = [row[0] for row in type_rows if row[1] == "numerical"]
        categorical_columns = [row[0] for row in type_rows if row[1] == "categorical"]

        return descriptions, numerical_columns, categorical_columns
        

    def validate_query(self, query: str) -> bool:
        """Validates the generated SQL query against the database schema and returns True if valid, False otherwise.

        Args:
            query (str): sql query to be validated.
            db_file (str, optional): sqlite database file to be used. Defaults to SQLITE_DB_FILE.

        Returns:
            bool: True if query is valid, False otherwise.
        """
        if not query.strip() or not query.lower().startswith("select"):
            return False

        try:
            match = re.search(r"FROM\s+(\w+)", query, re.IGNORECASE)
            if not match:
                return False
            table_name = match.group(1).strip()

            column_match = re.search(r"SELECT\s+(.+?)\s+FROM", query, re.IGNORECASE)
            if not column_match:
                return False
            columns = column_match.group(1).strip().split(",")

            self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            if not self.cursor.fetchone():
                return False

            self.cursor.execute(f"PRAGMA table_info({table_name})")
            table_columns = [column[1] for column in self.cursor.fetchall()]
            for column in columns:
                column = column.strip()
                if column not in table_columns and column != "*":
                    return False

            return True
        except sqlite3.Error as e:
            logger.error(f"Error validating query: {e}")
            return False


    def fetch_data_from_sqlite(self, table_name: str) -> pd.DataFrame:
        """Fetch data from a SQLite database table.

        Args:
            db_file (Path): Path to the SQLite database file.
            table_name (str): Name of the table to fetch data from.

        Returns:
            Optional[pd.DataFrame]: A DataFrame containing the data from the table, or None if an error occurred.
        """
        try:
            conn = sqlite3.connect(self.db_file)
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, conn)
            conn.close()
        except sqlite3.Error as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()  # Return an empty DataFrame on error

        return df
