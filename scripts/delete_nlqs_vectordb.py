import os

import pandas as pd

from nlqs import vectordb_driver
from nlqs.database.postgres import PostgresConnectionConfig
from nlqs.nlqs import NLQS
from nlqs.tests.conftest import chroma_config
from nlqs.vectordb_driver import ChromaDBConfig, VectorDBDriver

chroma_config = ChromaDBConfig(
    host=os.getenv("VECTORDB_HOST", "localhost"),
    port=int(os.getenv("VECTORDB_PORT", 8000)),
    is_local=False,
    username=os.getenv("VECTORDB_USER", "admin"),
    password=os.getenv("VECTORDB_PASSWORD", "potter"),
)


vectordb_driver = VectorDBDriver(chroma_config)


# Delete the NLQS VectorDB Collections
