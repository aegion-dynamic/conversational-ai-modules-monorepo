from typing import Dict, List
from chromadb import Collection, QueryResult


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


def construct_descriptive_search_query_fragments(lookup_dict: Dict[str, List[str]]) -> List[str]:
    """Creates an SQL query from a dictionary of descriptive data.

    Args:
        lookup_dict (Dict[str, List[str]]): A dictionary of descriptive data in the form {'column_name': ['value1', 'value2', ...]}.

    Returns:
        List[str]: The generated SQL query fragments.
    """
    if not lookup_dict:
        return []  # Return an empty list if the dictionary is empty

    query_parts = []
    for column, values in lookup_dict.items():
        # Construct the query part
        values_list = ", ".join(f"'{value}'" for value in values)
        query_part = f"{column} IN ({values_list})"
        query_parts.append(query_part)

    return query_parts


def qualitative_search(collection: Collection, data: Dict[str, str], primary_key: str) -> List[int]:
    """Performs a similarity search on the database and returns up to 5 similar results per column.

    Args:
        collection (chromadb.Collection): The ChromaDB collection to search.
        data (Dict[str, str]): A dictionary of qualitative data to search for.
        primary_key (str): The primary key column name in the database.

    Returns:
        List[int]: A list of unique IDs from the search results.
    """
    ids_per_column = {}

    for column, condition in data.items():
        query_result: QueryResult = collection.query(query_texts=condition, n_results=5, where={"column_name": column})

        if query_result["metadatas"]:
            ids_for_column = set()
            for result in query_result["metadatas"]:
                for item in result:
                    id_value = item.get(primary_key)
                    if id_value is not None:
                        ids_for_column.add(int(id_value))
            ids_per_column[column] = list(ids_for_column)

    print(f"ids_per_column: {ids_per_column}")

    # Flatten the list of lists into a single list of unique IDs
    all_ids = list(set([id_val for sublist in ids_per_column.values() for id_val in sublist]))
    return all_ids
