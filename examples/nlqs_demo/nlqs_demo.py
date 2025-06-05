import os
from pathlib import Path
import json
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Validate required environment variables
required_vars = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT",
]

missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

from nlqs.database.sqlite import SQLiteConnectionConfig, SQLiteDriver
from nlqs.vectordb_driver import ChromaDBConfig
from nlqs.nlqs import NLQS, NLQSResult

# Setup database configuration
db_path = Path("aegion.db")  # Using the existing database
sqlite_config = SQLiteConnectionConfig(
    db_file=db_path,
    dataset_table_name="new_dataset",  # Using the cannabis product dataset
    uri_column="URL",  # URL column contains the links
    output_columns=["Product", "Category", "CBD", "THC", "Description", "MedicalBenefitsReported"]  # Main columns of interest
)

# Setup ChromaDB configuration
chroma_config = ChromaDBConfig(
    persist_path=Path("chroma"),  # Using the existing chroma directory
    dataset_collection_name="nlqs_descriptive_data"  # This should match your existing collection
)

# Environment variables should already be loaded from .env file
os.environ["AZURE_OPENAI_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

def print_result(result: Dict[str, Any]) -> None:
    """Print a single result record in a formatted way."""
    print("\nProduct Details:")
    print("-" * 50)
    for key, value in result.items():
        if value:  # Only print non-empty values
            print(f"{key.replace('_', ' ').title()}: {value}")

class NLQSDemo:
    def __init__(self):
        """Initialize the NLQS demo with configuration."""
        self.nlqs = NLQS(connection_config=sqlite_config, chroma_config=chroma_config)
        self.chat_history = []  # Initialize empty chat history

    def process_query(self, query: str) -> None:
        """Process a natural language query and print results."""
        print(f"\nQuery: {query}")
        print("=" * 50)
        
        try:
            result = self.nlqs.execute_nlqs_query_workflow(query, self.chat_history)
            
            if result.is_input_irrelevant:
                print("\n⚠️ Query was flagged as irrelevant (possible SQL injection or general conversation)")
                return
            
            if not result.records:
                print("\n❌ No matching records found")
                return
            
            print(f"\n✅ Found {len(result.records)} matching records:")
            for record in result.records:
                print_result(record)
            
            if result.uris:
                print("\nAssociated URLs:")
                print("-" * 50)
                for uri in result.uris:
                    print(f"🔗 {uri}")
        
        except Exception as e:
            print(f"\n❌ Error processing query: {str(e)}")
            
def main():
    """Run the NLQS demo with test queries."""
    # Initialize NLQS demo
    demo = NLQSDemo()

    # Test queries covering different aspects
    test_queries = [
        # Product queries
        "Show me products with high CBD content",
        "Find products with both CBD and THC content above 15%",
        "What are the strongest indica products available?",
        
        # Medical benefit queries
        "Find products that help with pain relief",
        "What products are good for anxiety and stress?",
        "Show me products that help with sleep issues",
        
        # Category-specific queries
        "What edible products are available?",
        "Show me all vape cartridges",
        "List available concentrates",
        
        # Combination queries
        "Find high CBD products that help with inflammation",
        "Show me indica strains good for sleep with THC over 20%",
        
        # Edge cases
        "Hello, how are you?",  # Test phatic communication
        "SELECT * FROM new_dataset;",  # Test SQL injection detection
        "",  # Test empty input
    ]
    
    print("Starting NLQS Demo...")
    print("=" * 50)
    
    for query in test_queries:
        demo.process_query(query)
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
