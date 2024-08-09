import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "replace with actual key")

from nlqs.database.sqlite import SQLiteConnectionConfig, SQLiteDriver
from nlqs.database.postgres import PostgresConnectionConfig
from nlqs.nlqs import ChromaDBConfig
from pathlib import Path

# SQLite configuration
sqlite_config = SQLiteConnectionConfig(
    db_file=Path("aegion.db"),  
    dataset_table_name="new_dataset"
)

# PostgreSQL configuration
postgres_config = PostgresConnectionConfig(
    host="localhost", 
    port=5432, 
    user="postgres",  
    password="password",
    database_name="aegion",  
    dataset_table_name="new_dataset"  
)

# ChromaDB configuration
chroma_config = ChromaDBConfig(
    collection_name="aegion",  
    persist_path=Path("./chroma")  
)

# Choose either sqlite_config or postgres_config based on your database type
connection_config = sqlite_config 
# Or connection_config = postgres_config

driver = SQLiteDriver(connection_config)
driver.connect()

data_fetched = driver.fetch_data_from_database(connection_config.dataset_table_name)

PRODUCT_DESCRIPTIONS_CSV = "./product_descriptions.csv"
