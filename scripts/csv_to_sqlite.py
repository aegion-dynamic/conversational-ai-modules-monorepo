"""
Used to convert the CSV file to a SQLite database table. This script is used in preparation for 
the API to query the SQLite database.
"""

import sqlite3

import pandas as pd

from nlqs.parameters import connection_config, driver, PRODUCT_DESCRIPTIONS_CSV

def convert_csv_to_sqlite():
    try:
        # Read the CSV file into a DataFrame
        # Attempt using UTF-8 encoding first
        try:
            df = pd.read_csv(PRODUCT_DESCRIPTIONS_CSV)
        except UnicodeDecodeError:
            # If UTF-8 fails, fall back to ISO-8859-1
            df = pd.read_csv(PRODUCT_DESCRIPTIONS_CSV, encoding="ISO-8859-1")

        # Connect to the SQLite database
        conn = sqlite3.connect(connection_config.db_file)

        # Write the DataFrame to a SQLite table
        df.to_sql(connection_config.dataset_table_name, conn, if_exists="replace", index=False)

        print(f"Data written to {connection_config.dataset_table_name} in {connection_config.db_file}")

        # Commit the changes
        conn.commit()

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Close the connection
        conn.close() # type: ignore
