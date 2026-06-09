from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Sequence

import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor

from parts_catalogue.models import Part


class PartRepository(Protocol):
    """Interface for loading catalogue part details from storage."""

    def fetch_parts(self, part_ids: Sequence[str]) -> Mapping[str, Part]:
        """Fetch parts by id.

        Args:
            part_ids (Sequence[str]): Part identifiers.

        Returns:
            Mapping[str, Part]: Parts keyed by id.
        """


@dataclass(frozen=True)
class PostgresPartsRepositoryConfig:
    """Connection and table settings for a Postgres-backed parts catalogue."""

    host: str
    port: int
    user: str
    password: str
    database_name: str
    table_name: str
    id_column: str = "id"
    schema_name: str = "public"
    name_column: Optional[str] = "name"
    description_column: Optional[str] = "description"
    attribute_columns: Optional[Sequence[str]] = None


class PostgresPartsRepository:
    """Loads part records from a Postgres catalogue table."""

    def __init__(
        self,
        config: PostgresPartsRepositoryConfig,
        connection_factory: Callable[..., Any] = psycopg2.connect,
    ):
        """Initialize the repository.

        Args:
            config (PostgresPartsRepositoryConfig): Postgres table configuration.
            connection_factory (Callable[..., Any]): Injectable connection factory.
        """

        self.config = config
        self.connection_factory = connection_factory

    def fetch_parts(self, part_ids: Sequence[str]) -> Mapping[str, Part]:
        """Fetch part records by id."""

        unique_part_ids = tuple(dict.fromkeys(str(part_id) for part_id in part_ids if part_id))
        if not unique_part_ids:
            return {}

        query = self._build_fetch_query()
        with closing(self._connect()) as connection:
            with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, (list(unique_part_ids),))
                rows = cursor.fetchall()

        return {str(row[self.config.id_column]): self._row_to_part(row) for row in rows}

    def _connect(self) -> Any:
        """Open a database connection."""

        return self.connection_factory(
            dbname=self.config.database_name,
            user=self.config.user,
            password=self.config.password,
            host=self.config.host,
            port=self.config.port,
        )

    def _build_fetch_query(self) -> sql.SQL:
        """Build the parameterized query for fetching parts."""

        columns = self._selected_columns()
        if columns is None:
            selected_columns = sql.SQL("*")
        else:
            selected_columns = sql.SQL(", ").join(sql.Identifier(column) for column in columns)

        return sql.SQL("SELECT {columns} FROM {schema}.{table} WHERE {id_column}::text = ANY(%s)").format(
            columns=selected_columns,
            schema=sql.Identifier(self.config.schema_name),
            table=sql.Identifier(self.config.table_name),
            id_column=sql.Identifier(self.config.id_column),
        )

    def _selected_columns(self) -> Optional[Sequence[str]]:
        """Return selected columns or None to select all columns."""

        if self.config.attribute_columns is None:
            return None

        columns = [self.config.id_column]
        for column in (self.config.name_column, self.config.description_column):
            if column:
                columns.append(column)
        columns.extend(self.config.attribute_columns)
        return tuple(dict.fromkeys(columns))

    def _row_to_part(self, row: Mapping[str, Any]) -> Part:
        """Convert a database row to a Part."""

        part_id = str(row[self.config.id_column])
        name = _optional_row_value(row, self.config.name_column)
        description = _optional_row_value(row, self.config.description_column)
        return Part(id=part_id, name=name, description=description, attributes=dict(row))


def _optional_row_value(row: Mapping[str, Any], column_name: Optional[str]) -> Optional[str]:
    """Read an optional string column from a row."""

    if column_name is None or column_name not in row or row[column_name] is None:
        return None
    return str(row[column_name])
