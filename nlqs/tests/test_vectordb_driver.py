from pathlib import Path
from random import sample

import pandas as pd
import pytest

from nlqs import vectordb_driver
from nlqs.vectordb_driver import ChromaDBConfig, ColumnType, VectorDBDriver


def test_initialize_nlqs_vectordb(chroma_config):
    # # Mock the create_collection method
    # chromadb.PersistentClient.create_collection = lambda name: True

    # VectorDBDriver.initialize_nlqs_vectordb(chroma_config)

    # # Check if collections are created
    # assert chromadb.PersistentClient.create_collection(chroma_config.column_info_collection_name) is True
    # assert chromadb.PersistentClient.create_collection(chroma_config.dataset_collection_name) is True
    raise NotImplementedError("Test not implemented")


def test_populate_nlqs_vectordb(chroma_config):
    # # Mock the add method
    # chromadb.PersistentClient.add = lambda ids, documents, embeddings, metadatas: True

    # VectorDBDriver.populate_nlqs_vectordb(chroma_config)

    # # Check if collections are populated
    # assert chromadb.PersistentClient.add(ids=["1"], documents=[], embeddings=[], metadatas=[]) is True
    raise NotImplementedError("Test not implemented")


def test_initialize_nlqs_collection(chroma_config: ChromaDBConfig):

    VectorDBDriver.initialize_nlqs_vectordb(chroma_config)
    driver = VectorDBDriver(chroma_config)

    # Check if collections are created
    assert driver.column_info_collection is not None
    assert driver.dataset_collection is not None
    assert driver.table_description_collection is not None


def test_populate_nlqs_column_info(chroma_config: ChromaDBConfig):

    VectorDBDriver.purge_nlqs_vectordb(chroma_config)
    VectorDBDriver.initialize_nlqs_vectordb(chroma_config)

    column_info_df = pd.read_csv("./nlqs/tests/data/column_descriptions_with_embeddings.tsv", sep="\t")

    VectorDBDriver.populate_nlqs_column_info(
        chroma_config,
        column_info_df,
    )

    vectordb_driver = VectorDBDriver(chroma_config)

    # Ensure that there are as many items as the number of rows in the dataframe
    assert column_info_df.shape[0] == vectordb_driver.column_info_collection.count()


def test_populate_nlqs_dataset_info(chroma_config: ChromaDBConfig):

    VectorDBDriver.purge_nlqs_vectordb(chroma_config)
    VectorDBDriver.initialize_nlqs_vectordb(chroma_config)

    data_info_df = pd.read_csv("./nlqs/tests/data/data_descriptions_with_embeddings.tsv", sep="\t")

    VectorDBDriver.populate_nlqs_dataset_info(
        chroma_config,
        data_info_df,
    )

    vectordb_driver = VectorDBDriver(chroma_config)

    # Ensure that there are as many items as the number of rows in the dataframe
    assert data_info_df.shape[0] == vectordb_driver.dataset_collection.count()


def test_populate_nlqs_table_info(chroma_config: ChromaDBConfig):

    VectorDBDriver.purge_nlqs_vectordb(chroma_config)
    VectorDBDriver.initialize_nlqs_vectordb(chroma_config)

    table_description_df = pd.read_csv("./nlqs/tests/data/table_descriptions_with_embeddings.tsv", sep="\t")

    VectorDBDriver.populate_nlqs_table_info(
        chroma_config,
        table_description_df,
    )

    vectordb_driver = VectorDBDriver(chroma_config)

    # Ensure that there are as many items as the number of rows in the dataframe
    assert table_description_df.shape[0] == vectordb_driver.table_description_collection.count()


def test_get_closest_column_from_description(vectordb_driver: VectorDBDriver):

    column_name, column_type = vectordb_driver.get_closest_column_from_description(
        approximate_column_name="Product Description",
        users_description="A column that describes the product in detail.",
        sample_data_strings=[],
    )

    assert column_name == "Description"
    assert column_type == ColumnType.DESCRIPTIVE
