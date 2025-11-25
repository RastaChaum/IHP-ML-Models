ARG BUILD_FROM=ghcr.io/home-assistant/amd64-base-debian:trixie
FROM ${BUILD_FROM}

# Set shell options
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install system dependencies for XGBoost (Debian base)
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        python3-venv \
        build-essential \
        libgomp1 \
        libopenblas-dev \
        jq \
    && rm -rf /var/lib/apt/lists/*

# Create and use a virtual environment to avoid PEP 668 restrictions
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/
RUN /opt/venv/bin/pip install --no-cache-dir --upgrade pip wheel setuptools \
    && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm -rf /tmp/requirements.txt

# Copy application code
COPY rootfs /

# Set working directory
WORKDIR /app

# Make run script executable
RUN chmod a+x /scripts/run.sh

# Expose HTTP API port
EXPOSE 5000

# Set entrypoint
CMD ["/scripts/run.sh"]
