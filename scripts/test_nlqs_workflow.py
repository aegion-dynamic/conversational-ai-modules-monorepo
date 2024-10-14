from nlqs.database.postgres import PostgresConnectionConfig
from nlqs.nlqs import NLQS, ChromaDBConfig

nlsq = NLQS(
    connection_config=PostgresConnectionConfig(
        host="aws-0-us-east-1.pooler.supabase.com",
        port=6543,
        user="postgres.xdvwtpqclkedpktjsrzc",
        password="aOoDlcdghQ39Gkjr",
        database_name="postgres",
        dataset_table_name="new_dataset",
        uri_column="URL",
    ),
    chroma_config=ChromaDBConfig(),
)

user_input = "Suggest me a few products for headaches ?"

response = nlsq.execute_nlqs_query_workflow(user_input, [])

print("Response:")
print(response)
