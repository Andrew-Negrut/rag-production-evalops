import pytest
from alembic import command
from alembic.config import Config


@pytest.fixture(scope="session", autouse=True)
def apply_migrations():
    """
    Apply DB migrations once for the whole test session.
    Uses the same DATABASE_URL as the test container environment.
    """
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")
