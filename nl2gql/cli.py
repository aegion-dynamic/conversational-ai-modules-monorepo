import argparse
from pathlib import Path
from dotenv import load_dotenv
import requests
from .chains import create_graphql_chain, run_chain
from rich import print

from .schema_utils import load_schema, ready_schema_file

load_dotenv()  # Loads your OPENAI_API_KEY from .env

GRAPHQL_ENDPOINT = "https://countries.trevorblades.com/"
SCHEMA_FILE_PATH = "schema.graphql"

def nl2gql(nl_query: str, schema_path: Path) -> str:
    """
    Convert natural language query to GraphQL query using the provided schema.
    
    :param nl_query: Natural language question to convert.
    :param schema: GraphQL schema as a string.
    :return: Generated GraphQL query as a string.
    """
    schema = load_schema(schema_path)  # Load schema from file
    chain = create_graphql_chain()
    gql_query = run_chain(chain, schema, nl_query)
    return gql_query


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", required=True, help="Path to .graphql schema")
    parser.add_argument("--query", required=True, help="Natural language question")

    args = parser.parse_args()

    gql_query = nl2gql(args.query, args.schema)

    print("\n🧠 Generated GraphQL Query:\n")
    print(gql_query)

if __name__ == "__main__":
    # main()
    ready_schema_file(SCHEMA_FILE_PATH)  # Ensure schema file is ready
    main()  # Run the main CLI function