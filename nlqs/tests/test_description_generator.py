from pathlib import Path
from nlqs.database.sqlite import SQLiteConnectionConfig
from nlqs.description_generator import get_chroma_collection


def test_get_chroma_collection(sqlite_driver):

    sqlite_config = SQLiteConnectionConfig(db_file=Path("aegion.db"), dataset_table_name="new_dataset")

    db_driver = sqlite_driver

    db_driver.connect()

    primary_key = db_driver.get_primary_key(db_driver.db_config.dataset_table_name)

    ret = get_chroma_collection(
        collection_name="test",
        db_driver=db_driver,
        dataset_table_name=sqlite_config.dataset_table_name,
        primary_key=primary_key,
    )
