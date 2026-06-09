from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Union

from nlqs.database.postgres import PostgresDriver
from nlqs.database.sqlite import SQLiteDriver

logger = logging.getLogger(__name__)


class SearchField:
    """This a class that captures the various search parameters, the
    generated queries and the results from the search.

    This allows us to encapsulate the complexity of the search into a single black box.

    The search follows the NLQS algorithm (see ``nlqs/README.md``):

    1. Each field type (descriptive, categorical, identifier, quantitative) is
       turned into a single ``SELECT <primary_key> FROM <table> WHERE ...`` query
       where the field's fragments are combined with ``AND``. This produces a set
       of candidate primary keys per field type.
    2. The candidate sets are intersected across field types to find the exact
       matches.
    3. If the intersection is empty, the union of the candidate sets is used as a
       "related" fallback. ``is_exact_match`` records which path produced the keys.
    """

    def __init__(
        self,
        descriptive_query_fragments: List[str],
        categorical_query_fragments: List[str],
        identifier_query_fragments: List[str],
        quantitative_query_fragments: List[str],
        database_driver: Union[PostgresDriver, SQLiteDriver],
        primary_key: str,
    ) -> None:
        self.descriptive_query_fragments: List[str] = descriptive_query_fragments
        self.categorical_query_fragments: List[str] = categorical_query_fragments
        self.identifier_query_fragments: List[str] = identifier_query_fragments
        self.quantitative_query_fragments: List[str] = quantitative_query_fragments
        self.database_driver: Union[PostgresDriver, SQLiteDriver] = database_driver
        self.primary_key: str = primary_key

        # Candidate primary key set for each constrained field type
        self.field_results: Dict[str, Set[Any]] = {}
        # Aggregated primary keys after intersection / union
        self.primary_keys: List[Any] = []
        # Whether the aggregated keys came from the intersection (exact) path
        self.is_exact_match: bool = True

    def _run_field_query(self, fragments: List[str], table_name: str) -> Optional[Set[Any]]:
        """Run a single field-type query and return the set of matching primary keys.

        The fragments are combined with ``AND`` so that a row must satisfy every
        constraint of that field type. Only the primary key column is selected so
        that the primary key can be harvested reliably regardless of column order.

        Args:
            fragments (List[str]): The ``WHERE`` fragments for this field type.
            table_name (str): The name of the table to query.

        Returns:
            Optional[Set[Any]]: The set of matching primary keys, or ``None`` if the
            field type is not constrained (no fragments).
        """
        if not fragments:
            return None

        where_clause = " AND ".join(fragments)
        query = f"SELECT {self.primary_key} FROM {table_name} WHERE {where_clause};"

        result = self.database_driver.execute_query(query)

        primary_keys: Set[Any] = set()
        if result:
            for row in result:
                if row:
                    primary_keys.add(row[0])

        return primary_keys

    def run_queries(self, table_name: str) -> None:
        """Runs the queries on the database and computes the aggregated primary keys.

        Args:
            table_name (str): the name of the table
        """

        field_pk_sets: Dict[str, Set[Any]] = {}
        for field_name, fragments in (
            ("descriptive", self.descriptive_query_fragments),
            ("categorical", self.categorical_query_fragments),
            ("identifier", self.identifier_query_fragments),
            ("quantitative", self.quantitative_query_fragments),
        ):
            primary_keys = self._run_field_query(fragments, table_name)
            if primary_keys is not None:
                field_pk_sets[field_name] = primary_keys

        self.field_results = field_pk_sets

        if not field_pk_sets:
            self.primary_keys = []
            self.is_exact_match = True
            return

        candidate_sets = list(field_pk_sets.values())

        # Exact match: rows that satisfy every constrained field type
        intersection = set.intersection(*candidate_sets)
        if intersection:
            self.primary_keys = list(intersection)
            self.is_exact_match = True
            logger.debug(f"Exact match found with {len(self.primary_keys)} primary keys")
            return

        # Fallback: rows that satisfy at least one field type ("related" results)
        union = set.union(*candidate_sets)
        self.primary_keys = list(union)
        self.is_exact_match = False
        logger.debug(f"No exact match; falling back to union with {len(self.primary_keys)} primary keys")

    def get_primary_keys(self) -> List[Any]:
        """Return the aggregated primary keys produced by the search."""
        return self.primary_keys

    def get_results(self) -> Dict[str, Set[Any]]:
        """Return the per-field-type candidate primary key sets."""
        return self.field_results

    @staticmethod
    def construct_search_field(
        descriptive_query_fragments: List[str],
        categorical_query_fragments: List[str],
        identifier_query_fragments: List[str],
        quantitative_query_fragments: List[str],
        database_driver: Union[PostgresDriver, SQLiteDriver],
        database_name: str,  # TODO: Support multiple databases later
        table_name: str,  # TODO: Support multiple tables later
        primary_key: str,
    ) -> SearchField:

        ret = SearchField(
            descriptive_query_fragments=descriptive_query_fragments,
            categorical_query_fragments=categorical_query_fragments,
            identifier_query_fragments=identifier_query_fragments,
            quantitative_query_fragments=quantitative_query_fragments,
            database_driver=database_driver,
            primary_key=primary_key,
        )

        ret.run_queries(table_name)

        return ret
