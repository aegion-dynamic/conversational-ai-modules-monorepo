"""Unit tests for the NLQS search aggregation and query construction logic.

These tests exercise the core algorithm pieces (SearchField intersection/union and
descriptive fragment quoting) without requiring a live database, vector store or
LLM.
"""

from typing import Any, List, Optional, Tuple

from nlqs.query_construction import construct_descriptive_search_query_fragments
from nlqs.search_field import SearchField


class FakeDriver:
    """A minimal database driver double that returns canned rows per query."""

    def __init__(self, responses: List[Tuple[str, List[Tuple[Any, ...]]]]):
        # responses: list of (substring_to_match, rows_to_return)
        self.responses = responses
        self.queries: List[str] = []

    def execute_query(self, query: str) -> Optional[List[Tuple[Any, ...]]]:
        self.queries.append(query)
        for substring, rows in self.responses:
            if substring in query:
                return rows
        return []


class FakeVectorDB:
    """A vector DB double returning a fixed qualitative search result."""

    def __init__(self, result):
        self._result = result

    def qualitative_dataset_search(self, data, db_name, table_name):
        return self._result


def _make_search_field(driver, **fragments) -> SearchField:
    return SearchField(
        descriptive_query_fragments=fragments.get("descriptive", []),
        categorical_query_fragments=fragments.get("categorical", []),
        identifier_query_fragments=fragments.get("identifier", []),
        quantitative_query_fragments=fragments.get("quantitative", []),
        database_driver=driver,
        primary_key=fragments.get("primary_key", "id"),
    )


def test_search_field_intersection_is_exact_match():
    driver = FakeDriver(
        [
            ("cat_col", [(1,), (2,), (3,)]),
            ("num_col", [(2,), (3,), (4,)]),
        ]
    )
    sf = _make_search_field(
        driver,
        categorical=["cat_col = 'x'"],
        quantitative=["num_col > 1"],
    )
    sf.run_queries("t")

    assert sf.is_exact_match is True
    assert set(sf.get_primary_keys()) == {2, 3}


def test_search_field_union_fallback_when_no_intersection():
    driver = FakeDriver(
        [
            ("cat_col", [(1,), (2,)]),
            ("num_col", [(3,), (4,)]),
        ]
    )
    sf = _make_search_field(
        driver,
        categorical=["cat_col = 'x'"],
        quantitative=["num_col > 1"],
    )
    sf.run_queries("t")

    assert sf.is_exact_match is False
    assert set(sf.get_primary_keys()) == {1, 2, 3, 4}


def test_search_field_single_field_returns_its_rows():
    driver = FakeDriver([("id IN", [(5,), (6,)])])
    sf = _make_search_field(driver, descriptive=["id IN (5,6)"])
    sf.run_queries("t")

    assert sf.is_exact_match is True
    assert set(sf.get_primary_keys()) == {5, 6}


def test_search_field_no_constraints_returns_empty():
    driver = FakeDriver([])
    sf = _make_search_field(driver)
    sf.run_queries("t")

    assert sf.get_primary_keys() == []
    assert sf.is_exact_match is True
    # No queries should be issued when there are no fragments.
    assert driver.queries == []


def test_search_field_selects_primary_key_explicitly():
    driver = FakeDriver([("cat", [(1,)])])
    sf = _make_search_field(driver, categorical=["cat = 'x'"], primary_key="my_pk")
    sf.run_queries("mytable")

    assert any(q.startswith("SELECT my_pk FROM mytable WHERE") for q in driver.queries)


def test_descriptive_fragments_quote_string_keys():
    fake = FakeVectorDB({"Description": [("PackageID", "ABC"), ("PackageID", "DEF")]})
    out = construct_descriptive_search_query_fragments({"Description": "creamy"}, fake)

    assert out["Description"] == ["PackageID IN ('ABC', 'DEF')"]


def test_descriptive_fragments_leave_numeric_keys_unquoted():
    fake = FakeVectorDB({"Description": [("id", "1"), ("id", "2")]})
    out = construct_descriptive_search_query_fragments({"Description": "creamy"}, fake)

    assert out["Description"] == ["id IN (1, 2)"]


def test_descriptive_fragments_escape_embedded_quotes():
    fake = FakeVectorDB({"Description": [("name", "O'Brien")]})
    out = construct_descriptive_search_query_fragments({"Description": "x"}, fake)

    assert out["Description"] == ["name IN ('O''Brien')"]
