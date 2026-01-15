FROM python:3.11-slim

WORKDIR /app

# Install deps + your package
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy package source
COPY app ./app
COPY alembic.ini ./
COPY alembic ./alembic

# Install as editable package (so imports work cleanly)
RUN pip install --no-cache-dir -e .

# Copy tests (not required for runtime, but useful for CI)
COPY tests ./tests

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
