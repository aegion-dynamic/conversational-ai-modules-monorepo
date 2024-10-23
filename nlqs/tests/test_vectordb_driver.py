import pandas as pd
from nlqs.parameters import DEFAULT_DB_NAME, DEFAULT_TABLE_NAME
from nlqs.vectordb_driver import ChromaDBConfig, ColumnType, VectorDBDriver


def test_initialize_nlqs_vectordb(chroma_config, embedding_function):

    VectorDBDriver.initialize_nlqs_vectordb(chroma_config)

    driver = VectorDBDriver(chroma_config, embedding_function)

    # Check if all the collections are created
    assert driver.column_info_collection is not None
    assert driver.dataset_collection is not None
    assert driver.table_description_collection is not None


def test_initialize_nlqs_collection(chroma_config: ChromaDBConfig, embedding_function):

    VectorDBDriver.initialize_nlqs_vectordb(chroma_config)
    driver = VectorDBDriver(chroma_config, embedding_function)

    # Check if collections are created
    assert driver.column_info_collection is not None
    assert driver.dataset_collection is not None
    assert driver.table_description_collection is not None


def test_populate_nlqs_column_info(chroma_config: ChromaDBConfig, embedding_function):

    VectorDBDriver.purge_nlqs_vectordb(chroma_config)
    VectorDBDriver.initialize_nlqs_vectordb(chroma_config)

    column_info_df = pd.read_csv("./nlqs/tests/data/column_descriptions_with_embeddings.tsv", sep="\t")

    VectorDBDriver.populate_nlqs_column_info(
        chroma_config,
        column_info_df,
    )

    vectordb_driver = VectorDBDriver(chroma_config, embedding_function)

    # Ensure that there are as many items as the number of rows in the dataframe
    assert column_info_df.shape[0] == vectordb_driver.column_info_collection.count()


def test_populate_nlqs_dataset_info(chroma_config: ChromaDBConfig, embedding_function):

    VectorDBDriver.purge_nlqs_vectordb(chroma_config)
    VectorDBDriver.initialize_nlqs_vectordb(chroma_config)

    data_info_df = pd.read_csv("./nlqs/tests/data/data_descriptions_with_embeddings.tsv", sep="\t")

    VectorDBDriver.populate_nlqs_dataset_info(
        chroma_config,
        data_info_df,
    )

    vectordb_driver = VectorDBDriver(chroma_config, embedding_function)

    # Ensure that there are as many items as the number of rows in the dataframe
    assert data_info_df.shape[0] == vectordb_driver.dataset_collection.count()


def test_populate_nlqs_table_info(chroma_config: ChromaDBConfig, embedding_function):

    VectorDBDriver.purge_nlqs_vectordb(chroma_config)
    VectorDBDriver.initialize_nlqs_vectordb(chroma_config)

    table_description_df = pd.read_csv("./nlqs/tests/data/table_descriptions_with_embeddings.tsv", sep="\t")

    VectorDBDriver.populate_nlqs_table_info(
        chroma_config,
        table_description_df,
    )

    vectordb_driver = VectorDBDriver(chroma_config, embedding_function)

    # Ensure that there are as many items as the number of rows in the dataframe
    assert table_description_df.shape[0] == vectordb_driver.table_description_collection.count()


def test_get_closest_column_from_description(vectordb_driver: VectorDBDriver):

    # Note: This test is dependent on the data in the data files
    column_name, column_type = vectordb_driver.get_closest_column_from_description(
        approximate_column_name="Product Description",
        users_description="A column that describes the product in detail.",
        sample_data_strings=[],
        database_name=DEFAULT_DB_NAME,
        table_name=DEFAULT_TABLE_NAME,
    )

    assert column_name == "Description"
    assert column_type == ColumnType.DESCRIPTIVE


def test_retrieve_descriptions_and_types_from_db(vectordb_driver: VectorDBDriver):

    # Test without filters
    results = vectordb_driver.retrieve_descriptions_and_types_from_db()

    assert results is not None
    assert len(results["column_descriptions"].keys()) > 0

    # Test with filters
    results = vectordb_driver.retrieve_descriptions_and_types_from_db(
        db_name_filter=DEFAULT_DB_NAME,
        table_name_filter=DEFAULT_TABLE_NAME,
    )

    assert results is not None
    assert len(results["column_descriptions"].keys()) > 0

    # Test with random filters that should not return any results
    results = vectordb_driver.retrieve_descriptions_and_types_from_db(
        db_name_filter="random_db",
        table_name_filter="random_table",
    )

    assert results is None


def test_check_if_column_name_exists(vectordb_driver: VectorDBDriver):

    # Test with a column that exists
    result = vectordb_driver.check_if_column_name_exists(
        column_name="Description",
        db_name=DEFAULT_DB_NAME,
        table_name=DEFAULT_TABLE_NAME,
    )

    assert result is True

    # Test with a column that does not exist
    result = vectordb_driver.check_if_column_name_exists(
        column_name="random_column",
        db_name=DEFAULT_DB_NAME,
        table_name=DEFAULT_TABLE_NAME,
    )

    assert result is False
