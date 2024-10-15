from typing import Dict, List
from chromadb import Collection, QueryResult


def generate_quantitaive_search_query(quantitaive_data: Dict[str, str], table_name: str, primary_key: str) -> str:
    """Creates an SQL query from a dictionary of quantitative data.

    Args:
        quantitaive_data (dict): A dictionary of quantitative data in the form {'column_name': 'condition'}.

    Returns:
        str: The generated SQL query.
    """
    if not quantitaive_data:
        return ""  # Return an empty string if the dictionary is empty

    query_parts = []
    for column, condition in quantitaive_data.items():
        # Handle different comparison operators
        if "<" in condition:
            operator = "<"
        elif ">" in condition:
            operator = ">"
        elif "<=" in condition:
            operator = "<="
        elif ">=" in condition:
            operator = ">="
        elif "=" in condition:
            operator = "="
        else:
            operator = "LIKE"  # Default to LIKE for other conditions

        # Extract the value from the condition
        value = condition.replace(operator, "").strip()

        # Construct the query part
        query_part = f"{column} {operator} {value}"
        query_parts.append(query_part)

    # Combine the query parts with AND
    query_constraints = " AND ".join(query_parts)

    query = f"select {primary_key} from {table_name} where {query_constraints}"
    return query


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
