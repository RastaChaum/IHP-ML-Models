ARG BUILD_FROM
FROM ${BUILD_FROM}

# Set shell options
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install system dependencies for XGBoost
RUN apk add --no-cache \
    python3-dev \
    py3-pip \
    build-base \
    libgomp \
    openblas-dev

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

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
