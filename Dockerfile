FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency list first for better cache use
COPY requirements.txt /app/requirements.txt

# Ensure pip is recent
RUN python -m pip install --upgrade pip setuptools wheel

# Install Python deps
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project
COPY . /app

# Expose only backend port
EXPOSE 8000

# Command to run FastAPI
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
