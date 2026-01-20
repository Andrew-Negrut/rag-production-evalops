FROM python:3.11-slim

WORKDIR /app

# Install deps + your package
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy package source
COPY app ./app
COPY alembic.ini ./
COPY alembic ./alembic

ARG INSTALL_EXTRAS=""
# Install as editable package (so imports work cleanly)
RUN if [ -n "$INSTALL_EXTRAS" ]; then \
        pip install --no-cache-dir -e ".[${INSTALL_EXTRAS}]"; \
    else \
        pip install --no-cache-dir -e .; \
    fi

# Copy tests (not required for runtime, but useful for CI)
COPY tests ./tests

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
