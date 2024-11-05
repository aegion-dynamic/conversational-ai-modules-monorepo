from __future__ import annotations

from typing import Any, Dict, List, Union

from nlqs.database.postgres import PostgresDriver
from nlqs.database.sqlite import SQLiteDriver
from nlqs.query_construction import construct_final_search_query


class SearchField:
    """This a class that captures the various search parameters, the
    generated queries and the results from the search.

    This allows us to encapsulate the complexity of the search into a single black box.
    """

    def __init__(
        self,
        descriptive_query_fragments: List[str],
        categorical_query_fragments: List[str],
        identifier_query_fragments: List[str],
        quantitative_query_fragments: List[str],
        database_driver: Union[PostgresDriver, SQLiteDriver],
    ) -> None:
        self.descriptive_query_fragments: List[str] = descriptive_query_fragments
        self.categorical_query_fragments: List[str] = categorical_query_fragments
        self.identifier_query_fragments: List[str] = identifier_query_fragments
        self.quantitative_query_fragments: List[str] = quantitative_query_fragments
        self.database_driver: Union[PostgresDriver, SQLiteDriver] = database_driver

        # Datastore for the return type of the search
        self.search_results: Dict[str, List[Any]] = {}

    def run_queries(self, database_name: str, table_name: str):
        """Runs the queries on the database and stores the results in the search_results attribute.

        Args:
            database_name (str): The name of the database
            table_name (str): the name of the table

        Returns:
            _type_: Full rows of data from the database
        """

        # Run the queries
        descriptive_queries = construct_final_search_query(self.descriptive_query_fragments, database_name, table_name)

        results = []
        for query in descriptive_queries:
            result = self.database_driver.execute_query(query)

            if result is not None:
                results.extend(result)

        # TODO: Update to include searches for the rest of the fields this is should also create a more elaborate tree
        # structure that has all the intersections of the results, etc.

        # self.search_results["categorical"] = self.database_driver.run_search_query(
        #     database_name, table_name, self.categorical_query_fragments
        # )
        # self.search_results["identifier"] = self.database_driver.run_search_query(
        #     database_name, table_name, self.identifier_query_fragments
        # )
        # self.search_results["quantitative"] = self.database_driver.run_search_query(
        #     database_name, table_name, self.quantitative_query_fragments
        # )

        return results

    def get_results(self):

        return self.search_results

    @staticmethod
    def construct_search_field(
        descriptive_query_fragments: List[str],
        categorical_query_fragments: List[str],
        identifier_query_fragments: List[str],
        quantitative_query_fragments: List[str],
        database_driver: Union[PostgresDriver, SQLiteDriver],
        database_name: str,  # TODO: Support multiple databases later
        table_name: str,  # TODO: Support multiple tables later
    ) -> SearchField:

        ret = SearchField(
            descriptive_query_fragments=descriptive_query_fragments,
            categorical_query_fragments=categorical_query_fragments,
            identifier_query_fragments=identifier_query_fragments,
            quantitative_query_fragments=quantitative_query_fragments,
            database_driver=database_driver,
        )

        results = ret.run_queries(database_name, table_name)

        ret.search_results["default"] = results

        return ret
