FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# Copy source code
COPY flashback/ ./flashback/
COPY tests/ ./tests/
COPY config/ ./config/
COPY examples/ ./examples/

# Create output directory
RUN mkdir -p output

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASHBACK_CONFIG_DIR=/app/config

# Default command
CMD ["flashback", "--help"]
