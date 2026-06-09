from __future__ import annotations

import os
from typing import Mapping, Optional

from parts_catalogue.repository import PostgresPartsRepositoryConfig


def load_postgres_repository_config(
    environment: Optional[Mapping[str, str]] = None,
    prefix: str = "PARTS_CATALOGUE_",
) -> PostgresPartsRepositoryConfig:
    """Load Postgres catalogue configuration from environment variables.

    Args:
        environment (Optional[Mapping[str, str]]): Environment mapping. Defaults to os.environ.
        prefix (str): Environment variable prefix.

    Returns:
        PostgresPartsRepositoryConfig: Repository configuration.

    Raises:
        ValueError: If required settings are missing.
    """

    env = environment or os.environ
    required_values = {
        "host": _read_required(env, f"{prefix}POSTGRES_HOST"),
        "port": _read_required(env, f"{prefix}POSTGRES_PORT"),
        "user": _read_required(env, f"{prefix}POSTGRES_USER"),
        "password": _read_required(env, f"{prefix}POSTGRES_PASSWORD"),
        "database_name": _read_required(env, f"{prefix}POSTGRES_DATABASE"),
        "table_name": _read_required(env, f"{prefix}PARTS_TABLE"),
    }

    return PostgresPartsRepositoryConfig(
        host=required_values["host"],
        port=int(required_values["port"]),
        user=required_values["user"],
        password=required_values["password"],
        database_name=required_values["database_name"],
        table_name=required_values["table_name"],
        id_column=env.get(f"{prefix}PART_ID_COLUMN", "id"),
        schema_name=env.get(f"{prefix}POSTGRES_SCHEMA", "public"),
        name_column=env.get(f"{prefix}PART_NAME_COLUMN", "name"),
        description_column=env.get(f"{prefix}PART_DESCRIPTION_COLUMN", "description"),
    )


def _read_required(environment: Mapping[str, str], key: str) -> str:
    """Read a required environment variable."""

    value = environment.get(key)
    if not value:
        raise ValueError(f"Missing required environment variable: {key}")
    return value
