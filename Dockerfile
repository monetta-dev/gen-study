FROM python:3.12-slim

ARG TYPST_VERSION=0.15.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        fonts-noto-cjk \
        texlive-full \
        xz-utils \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    case "$arch" in \
        amd64) typst_arch="x86_64-unknown-linux-musl" ;; \
        arm64) typst_arch="aarch64-unknown-linux-musl" ;; \
        armhf) typst_arch="armv7-unknown-linux-musleabi" ;; \
        *) echo "Unsupported architecture for Typst: $arch" >&2; exit 1 ;; \
    esac; \
    curl -fsSL \
        "https://github.com/typst/typst/releases/download/v${TYPST_VERSION}/typst-${typst_arch}.tar.xz" \
        -o /tmp/typst.tar.xz; \
    mkdir -p /tmp/typst; \
    tar -xJf /tmp/typst.tar.xz -C /tmp/typst --strip-components=1; \
    install -m 0755 /tmp/typst/typst /usr/local/bin/typst; \
    rm -rf /tmp/typst /tmp/typst.tar.xz; \
    typst --version

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY mathgen ./mathgen
COPY viewer ./viewer
COPY README.md Specification.md Procfile ./

ENV API_HOST=0.0.0.0 \
    API_PORT=8000

CMD ["python3", "-m", "mathgen.api"]
