"""
Used to convert the CSV file to a SQLite database table. This script is used in preparation for 
the API to query the SQLite database.
"""

import pandas as pd
from pathlib import Path

from nlqs.database.sqlite import SQLiteConnectionConfig, SQLiteDriver


PRODUCT_DESCRIPTIONS_CSV = "../product_descriptions.csv"

# Read the CSV file into a DataFrame
# Attempt using UTF-8 encoding first
try:
    df = pd.read_csv(PRODUCT_DESCRIPTIONS_CSV)
except UnicodeDecodeError:
    # If UTF-8 fails, fall back to ISO-8859-1
    df = pd.read_csv(PRODUCT_DESCRIPTIONS_CSV, encoding="ISO-8859-1")

# Create a new column with unique values for primary key
primary_key_column = "id"
df[primary_key_column] = range(1, 1 + len(df))

# Move the primary key column to the first position
cols = df.columns.tolist()
cols.insert(0, cols.pop(cols.index(primary_key_column)))
df = df[cols]

print(df.info())

# Connect to the SQLite database
driver = SQLiteDriver(SQLiteConnectionConfig(db_file=Path("../aegion.db"), dataset_table_name="new_dataset"))

driver.connect()

conn = driver._db_connection

# Write the DataFrame to a SQLite table
df.to_sql(driver.db_config.dataset_table_name, conn, if_exists="replace", index=False)

print(f"Data written to {driver.db_config.dataset_table_name} in {driver.db_config.db_file}")

# Commit the changes
conn.commit()
conn.close()
