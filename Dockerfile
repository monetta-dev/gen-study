FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        texlive-full \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY mathgen ./mathgen
COPY viewer ./viewer
COPY README.md Specification.md Procfile ./

ENV API_HOST=0.0.0.0 \
    API_PORT=8000

CMD ["python3", "-m", "mathgen.api"]
