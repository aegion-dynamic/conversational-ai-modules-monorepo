import chromadb
from chromadb.config import Settings
from openai import OpenAI
import json

openai_client = OpenAI(api_key="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")
from expert_system.parameters import (
    VECTORDB_HOST,
    VECTORDB_PASSWORD,
    VECTORDB_PORT,
    VECTORDB_USERNAME,
)


# Initialize ChromaDB client
client = chromadb.HttpClient(
    host=VECTORDB_HOST,
    port=int(VECTORDB_PORT),
    settings=Settings(
        chroma_client_auth_provider="chromadb.auth.basic.BasicAuthClientProvider",
        chroma_client_auth_credentials=f"{VECTORDB_USERNAME}:{VECTORDB_PASSWORD}",
    ),
)

collections = [col.name for col in client.list_collections()]
print(f"collections in the chromdb were {collections}")

# Get the collection
# collection = client.get_collection(name="ced-library")  # collection name

# Fetch all embeddings and their metadata
# print(f"Count: {collection.count()}")
# results = collection.peek()

# results = collection.query(
#     query_texts=["what is cannabis"],
#     n_results=3,
#     # where={"metadata_field": "is_equal_to_this"},
#     # where_document={"$contains":"search_string"}
# )

# Function to get embeddings from OpenAI
# def get_openai_embeddings(texts):
#     response = openai_client.embeddings.create(input=texts,
#     model="text-embedding-ada-002")
#     return response.data[0].embedding

# # Example usage
# query_texts = ["what is cannabis"]
# embeddings = get_openai_embeddings(query_texts)

# # Use the embeddings in the query
# results = collection.query(
#     query_embeddings=embeddings,
#     n_results=3,
# )
# # Print the data
# print(results)

# # Write the data to a json file
# with open("data.json", "w") as f:
#     f.write(json.dumps(results, indent=2))

# client.delete_collection(name="aegion1")
# client.delete_collection(name="aegion")
