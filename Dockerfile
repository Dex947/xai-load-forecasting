FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (gcc for compilation, libgomp for LightGBM OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY config/ config/
COPY models/ models/

# Expose API port
EXPOSE 8000

# Run API server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
