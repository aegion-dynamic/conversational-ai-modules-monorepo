from typing import Dict, List
from unittest.mock import DEFAULT

from nlqs.parameters import DEFAULT_DB_NAME, DEFAULT_TABLE_NAME
from nlqs.vectordb_driver import VectorDBDriver


def join_fragments(fragments: List[str], joiner: str = "AND") -> str:
    """Joins a list of query fragments into a single query.

    Args:
        fragments (List[str]): A list of query fragments to join.
        joiner (str, optional): The joiner to use between fragments. Defaults to "AND".

    Returns:
        str: The joined query.
    """
    return f" {joiner} ".join(fragments)


def construct_quantitaive_search_query_fragments(quantitaive_data: Dict[str, str]) -> List[str]:
    """Creates an SQL query from a dictionary of quantitative data.

    Args:
        quantitaive_data (dict): A dictionary of quantitative data in the form {'column_name': 'condition'}.

    Returns:
        str: The generated SQL query fragment.
    """
    if not quantitaive_data:
        return []  # Return an empty string if the dictionary is empty

    query_parts = []
    for column, condition in quantitaive_data.items():

        # Remove the whitespace from the condition
        condition = condition.replace(" ", "")

        # Handle different comparison operators
        if "<=" in condition:
            operator = "<="
        elif ">=" in condition:
            operator = ">="
        elif "<" in condition:
            operator = "<"
        elif ">" in condition:
            operator = ">"
        elif "=" in condition:
            operator = "="
        else:
            print(f"Warning ! : Invalid condition: {condition}")
            continue

        # Extract the value from the condition
        value = condition.replace(operator, "").strip()

        # Construct the query part
        query_part = f"{column} {operator} {value}"
        query_parts.append(query_part)

    return query_parts


def construct_categorical_search_query_fragments(categorical_data: Dict[str, str]) -> List[str]:
    """Creates an SQL query from a dictionary of categorical data.

    Args:
        categorical_data (dict): A dictionary of categorical data in the form {'column_name': 'condition'}.

    Returns:
        str: The generated SQL query fragment.
    """
    if not categorical_data:
        return []  # Return an empty string if the dictionary is empty

    query_parts = []
    for column, condition in categorical_data.items():
        # Construct the query part
        query_part = f"{column} = '{condition}'"
        query_parts.append(query_part)

    # Combine the query parts with AND
    return query_parts


def construct_identifier_search_query_fragments(identifier_data: Dict[str, str]) -> List[str]:
    """Creates an SQL query from a dictionary of identifier data.

    Args:
        identifier_data (dict): A dictionary of identifier data in the form {'column_name': 'condition'}.

    Returns:
        str: The generated SQL query fragment.
    """
    if not identifier_data:
        return []  # Return an empty string if the dictionary is empty

    query_parts = []
    for column, condition in identifier_data.items():
        # Construct the query part
        query_part = f"{column} = {condition}"
        query_parts.append(query_part)

    # Combine the query parts with AND
    return query_parts


def construct_descriptive_search_query_fragments(
    descriptive_data: Dict[str, str], vectordb_driver: VectorDBDriver
) -> Dict[str, List[str]]:
    """Creates an SQL query from a dictionary of descriptive data.

    Args:
        descriptive_data (Dict[str, str]):  A dictionary of descriptive data in the form {'column_name': 'condition'}.

    Returns:
        Dict[str, List[str]]: A dictionary of the generated SQL query fragments where the key is the column name.
    """

    resutls = vectordb_driver.qualitative_dataset_search(
        data=descriptive_data, db_name=DEFAULT_DB_NAME, table_name=DEFAULT_TABLE_NAME
    )

    if not resutls:
        return {}  # Return an empty string if the dictionary is empty

    ret = {}

    for column, pk_column_name_value_pairs in resutls.items():
        query_parts = []
        # Construct a dictionary of primary key column names and values
        temp_storage: Dict[str, List[str]] = {}

        # Store the value_pairs in the temp_storage
        for pk_column_name, value in pk_column_name_value_pairs:
            if pk_column_name not in temp_storage:
                temp_storage[pk_column_name] = []
            temp_storage[pk_column_name].append(value)

        # Construct the query part for each primary key column
        for pk_column_name, values in temp_storage.items():
            values_list = ", ".join(f"{value}" for value in values)
            query_part = f"{pk_column_name} IN ({values_list})"
            query_parts.append(query_part)

        ret[column] = query_parts

    return ret


def construct_final_search_query(where_query_fragments: List[str], table_name: str) -> List[str]:
    """Construct the search query using the fragments (database and table names)

    Args:
        where_query_fragments (List[str]): The where conditions that need to be appended
        database_name (str): The name of the database
        table_name (str): The name fo the table

    Returns:
        List[str]: The list of queries
    """

    # Construct the final query
    if not where_query_fragments:
        return []

    queries = [f"SELECT * FROM {table_name} WHERE {fragment};" for fragment in where_query_fragments]
    return queries
