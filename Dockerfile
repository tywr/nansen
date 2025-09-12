FROM python:3.12-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y git

RUN uv pip install --system setuptools wheel twine build

WORKDIR /app

COPY ./pyproject.toml .
COPY ./setup.py .

RUN uv pip install --system -r pyproject.toml
