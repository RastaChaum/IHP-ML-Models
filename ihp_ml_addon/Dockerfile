ARG BUILD_FROM=ghcr.io/home-assistant/amd64-base-debian:trixie
FROM ${BUILD_FROM}

# Set shell options
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install system dependencies for XGBoost (Debian base)
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        build-essential \
        libgomp1 \
        libopenblas-dev \
        jq \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry and configure it
ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR='/var/cache/pypoetry' \
    POETRY_HOME='/usr/local'
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

# Copy ONLY dependency files first (for better caching)
WORKDIR /app
COPY pyproject.toml poetry.lock* /app/

# Install dependencies (this layer will be cached unless dependencies change)
RUN poetry install --without dev,test --no-root --no-interaction --no-ansi

# Copy S6 service definitions (these change rarely)
COPY ./ihp_ml_addon/rootfs/etc /etc

# Make scripts executable
RUN chmod a+x /etc/services.d/ihp-ml/run \
    && chmod a+x /etc/services.d/ihp-ml/finish

# Copy application code LAST (changes frequently in dev)
# Use wildcard to copy contents without replacing the directory itself
COPY ./ihp_ml_addon/rootfs/app/ /app/

# Expose HTTP API port
EXPOSE 5000

# S6 overlay will automatically start services in /etc/services.d/
# No need for explicit CMD - the base image handles it
