import pytest
from pathlib import Path
import chromadb

from nlqs.vectordb_driver import (
    ChromaDBConfig,
    ColumnType,
    VectorDBDriver,
)


def test_vectordb_driver_initialization(vectordb_driver):
    assert vectordb_driver.chroma_config.column_info_collection_name == "test_column_info"
    assert vectordb_driver.chroma_config.dataset_collection_name == "test_dataset_info"
    assert vectordb_driver.chroma_config.persist_path == Path("./test_chroma")
    assert vectordb_driver.chroma_config.host == "localhost"
    assert vectordb_driver.chroma_config.port == 8000
    assert vectordb_driver.chroma_config.is_local is True

def test_check_nlqs_collections_exists(vectordb_driver):
    # Mock the get_chroma_collection method
    vectordb_driver.get_chroma_collection = lambda name: True
    assert vectordb_driver.check_nlqs_collections_exists() is True

    vectordb_driver.get_chroma_collection = lambda name: None
    assert vectordb_driver.check_nlqs_collections_exists() is False

def test_get_chroma_collection(vectordb_driver):
    # Mock the chroma_client.get_collection method
    vectordb_driver.chroma_client.get_collection = lambda name: True
    assert vectordb_driver.get_chroma_collection("test_collection") is True

    vectordb_driver.chroma_client.get_collection = lambda name: None
    assert vectordb_driver.get_chroma_collection("test_collection") is None

def test_column_info_collection(vectordb_driver):
    # Mock the get_chroma_collection method
    vectordb_driver.get_chroma_collection = lambda name: True
    assert vectordb_driver.column_info_collection is True

    vectordb_driver.get_chroma_collection = lambda name: None
    with pytest.raises(ValueError):
        vectordb_driver.column_info_collection

def test_dataset_collection(vectordb_driver):
    # Mock the get_chroma_collection method
    vectordb_driver.get_chroma_collection = lambda name: True
    assert vectordb_driver.dataset_collection is True

    vectordb_driver.get_chroma_collection = lambda name: None
    with pytest.raises(ValueError):
        vectordb_driver.dataset_collection

def test_get_closest_column_from_description(vectordb_driver: VectorDBDriver):
    # Mock the column_info_collection property
    # vectordb_driver.column_info_collection = lambda: True

    # # Mock the query method
    # vectordb_driver.column_info_collection.query = lambda query_texts, n_results: {
    #     'metadatas': [[{"column_name": "test_column"}]]
    # }

    closest_column_name, column_type = vectordb_driver.get_closest_column_from_description(
        "approximate_column_name",
        "user_description",
        ["sample_data"]
    )

    assert closest_column_name == "test_column"
    assert column_type == ColumnType.DESCRIPTIVE

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