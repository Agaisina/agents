FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    gcc build-essential libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry

WORKDIR /app

COPY pyproject.toml poetry.lock* ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-root --no-interaction --no-ansi

RUN useradd -m -u 1000 user && chown -R user:user /app

USER user

ENV PATH="/home/user/.local/bin:${PATH}"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app"

COPY --chown=user:user . .

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "7860"]