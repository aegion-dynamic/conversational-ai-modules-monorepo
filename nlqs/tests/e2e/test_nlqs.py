from nlqs.database.postgres import PostgresConnectionConfig
from nlqs.nlqs import NLQS, ChromaDBConfig


def test_nlsq_api(chroma_config):
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
        chroma_config=chroma_config,
    )

    user_input = "suggest me a product for a headache"

    response = nlsq.execute_nlqs_query_workflow(user_input, [])

    print("Response:")
    print(response)
