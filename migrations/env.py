"""Alembic environment — targets SQLModel.metadata; URL from DATABASE_URL."""

import contextlib
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlmodel import SQLModel, create_engine

# Import the modules that define our tables so they register on SQLModel.metadata.
import effective.ledger  # noqa: F401
from effective.ledger import to_sqlalchemy_url

config = context.config
if config.config_file_name is not None:
    with contextlib.suppress(Exception):
        fileConfig(config.config_file_name)

target_metadata = SQLModel.metadata


def _url() -> str:
    return to_sqlalchemy_url(
        os.environ.get("DATABASE_URL", "postgresql://effective:effective@localhost:5432/effective")
    )


def run_migrations_online() -> None:
    connectable = create_engine(_url(), poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
