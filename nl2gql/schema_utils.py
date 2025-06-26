import requests


def load_schema(path: str) -> str:
    with open(path, "r") as f:
        return f.read().strip()


def ready_schema_file(filename: str = "schema.graphql"):
    try:
        with open(filename, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Schema file '{filename}' not found. Fetching schema...")
        schema = fetch_and_prepare_schema(filename)
        print("\n📂 Saving schema to file:", filename)
        create_schema_file(schema, filename)
        return schema


def fetch_graphql_schema(endpoint):
    introspection_query = {
        "query": """
        query IntrospectionQuery {
            __schema {
                types {
                    name
                    fields {
                        name
                        type {
                            name
                            kind
                            ofType {
                                name
                                kind
                            }
                        }
                    }
                }
            }
        }
        """
    }
    response = requests.post(endpoint, json=introspection_query)
    return response.json()


def fetch_and_prepare_schema(endpoint: str):
    full_schema = fetch_graphql_schema(endpoint)

    print("\n🗺️  Fetched GraphQL Schema:\n")

    # Extract a simple view of schema for the LLM (simplified for readability)
    relevant_types = [t for t in full_schema["data"]["__schema"]["types"] if t["fields"]]
    snippet = "\n".join(
        [
            f"type {t['name']} {{ " + ", ".join(f["name"] for f in t["fields"]) + " }}"
            for t in relevant_types
            if t["name"] in ["Query", "Country", "Language"]
        ]
    )
    return snippet


def create_schema_file(schema: str, filename: str = "schema.graphql"):
    with open(filename, "w") as f:
        f.write(schema)
