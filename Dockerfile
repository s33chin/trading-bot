FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy bot code
COPY . .

# Prometheus metrics port
EXPOSE 8000

# Default: paper trading
ENV TRADING_MODE=paper

ENTRYPOINT ["python", "main.py"]
