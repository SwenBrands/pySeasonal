FROM python:3.12 AS base

WORKDIR /build

# Copy only necessary files for installation (thanks to .dockerignore)
COPY pyproject.toml README.md ./
COPY pyseasonal/ ./pyseasonal/
COPY config/ ./config/

# Install package with all dependencies in a virtual environment
RUN python -m venv /opt/pyseasonal && \
    /opt/pyseasonal/bin/pip install --upgrade pip setuptools wheel && \
    /opt/pyseasonal/bin/pip install . && \
    /opt/pyseasonal/bin/pip cache purge

FROM python:3.12-slim

WORKDIR /app

# Copy the virtual environment from build stage
COPY --from=base /opt/pyseasonal /opt/pyseasonal

# Copy CLI scripts to /app for direct execution
COPY --from=base /build/pyseasonal/cli_*.py /app/scripts/

# Set PATH to include the virtual environment binaries
ENV PATH="/opt/pyseasonal/bin:$PATH"

ENTRYPOINT ["/opt/pyseasonal/bin/python"]

