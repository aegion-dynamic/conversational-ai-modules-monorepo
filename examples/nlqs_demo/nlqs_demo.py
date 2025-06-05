import os
from pathlib import Path

from nlqs.database.sqlite import SQLiteConnectionConfig, SQLiteDriver
from nlqs.vectordb_driver import ChromaDBConfig
from nlqs.nlqs import NLQS
from utils.parameters import (
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT_NAME
)

# Setup database configuration
db_path = Path("aegion.db")  # Using the existing database
sqlite_config = SQLiteConnectionConfig(
    db_file=db_path,    dataset_table_name="new_dataset",  # Using the cannabis product dataset
    uri_column="URL",  # URL column contains the links
    output_columns=["Product", "Category", "CBD", "THC", "Description", "MedicalBenefitsReported"]  # Main columns of interest
)

# Setup ChromaDB configuration
chroma_config = ChromaDBConfig(
    persist_path=Path("chroma"),  # Using the existing chroma directory
    dataset_collection_name="nlqs_descriptive_data"  # This should match your existing collection
)

# Set up environment variables for Azure OpenAI
os.environ["AZURE_OPENAI_API_KEY"] = "3mlWf94q4pwqLeSeBvebIXiiWQWM5ieFGLXfFXMwp4vkENm9UkSqJQQJ99ALACHYHv6XJ3w3AAAAACOGLDqM"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://ai-medicalbots438939302111.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"
os.environ["AZURE_OPENAI_DEPLOYMENT"] = "gpt-4o"
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-02-01"

# Initialize NLQS with Azure OpenAI
nlqs = NLQS(connection_config=sqlite_config, chroma_config=chroma_config)

# Example chat history (empty for now)
chat_history = []

def process_query(query: str):
    """Process a natural language query and print results."""
    print(f"\nQuery: {query}")
    print("-" * 50)
    
    result = nlqs.execute_nlqs_query_workflow(query, chat_history)
    
    if result.is_input_irrelevant:
        print("Query was flagged as irrelevant (possible SQL injection or general conversation)")
        return
    
    if not result.records:
        print("No matching records found")
        return
    
    print(f"Found {len(result.records)} matching records:")
    for record in result.records:
        print(record)
    
    if result.uris:
        print("\nAssociated URIs:")
        for uri in result.uris:
            print(uri)

if __name__ == "__main__":    # Test queries
    test_queries = [
        "Show me products with high CBD content",  # Descriptive query
        "Find products that help with pain relief",  # Medical benefits query
        "What cannabis products are available in Room A?",  # Location-based query
        "Hello, how are you?",  # Test phatic communication
        "SELECT * FROM new_dataset;",  # Test SQL injection detection
    ]
    
    for query in test_queries:
        process_query(query)
        print("\n" + "="*60 + "\n")
