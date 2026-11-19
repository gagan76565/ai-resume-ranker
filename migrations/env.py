"""Alembic environment.

When the app runs migrations it passes an open connection through
``config.attributes["connection"]``. When run from the command line
(``alembic upgrade head``) the URL comes from ``sqlalchemy.url`` or from the
same environment variables the app uses.
"""

from alembic import context

from talentlens.config import load_settings
from talentlens.db import make_engine, metadata

config = context.config
target_metadata = metadata


def run_migrations_offline():
    url = config.get_main_option("sqlalchemy.url") or load_settings().database_url
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    connection = config.attributes.get("connection")
    if connection is not None:
        context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=True)
        with context.begin_transaction():
            context.run_migrations()
        return
    url = config.get_main_option("sqlalchemy.url") or load_settings().database_url
    engine = make_engine(url)
    with engine.begin() as conn:
        context.configure(connection=conn, target_metadata=target_metadata, render_as_batch=True)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
