

import pandas as pd
from nlqs.database.postgres import PostgresConnectionConfig
from nlqs.nlqs import NLQS
from nlqs.vectordb_driver import ChromaDBConfig, VectorDBDriver

# Setup the NLQS VectorDB Collections
VectorDBDriver.initialize_nlqs_vectordb(ChromaDBConfig())


# Figure out how to populate the NLQS VectorDB Collections
#TODO: Populate the NLQS VectorDB column info collection
# Load the column info csv file
column_info_df = pd.read_csv("column_metadata.csv")
VectorDBDriver.populate_nlqs_vectordb(ChromaDBConfig(), column_info_df)

#TODO: Populate the NLQS VectorDB dataset collection
dataset_info_df = pd.read_csv("dataset.csv")
VectorDBDriver.populate_nlqs_vectordb(ChromaDBConfig(), dataset_info_df)
