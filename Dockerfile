FROM python:3.12 AS base

COPY . .

RUN python -m venv /opt/pyseasonal && \
    /opt/pyseasonal/bin/pip install . && \
    /opt/pyseasonal/bin/pip cache purge

FROM python:3.12-slim

WORKDIR /app
COPY --from=base /opt/pyseasonal /opt/pyseasonal

ENV PATH="/opt/pyseasonal/bin:$PATH"

COPY pyseasonal/ /app/pyseasonal/

ENTRYPOINT ["/opt/pyseasonal/bin/python"]

