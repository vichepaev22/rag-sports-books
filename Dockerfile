# Deploy to Railway (free tier)
# 1. Push this folder to GitHub
# 2. Railway → New Project → Deploy from GitHub repo
# 3. Add env var: NVIDIA_API_KEY
# 4. Done

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY rag_engine.py server.py ./

# Copy skill data (read-only, immutable SQLite)
COPY /home/hermes/.hermes/skills/books/physiology-sport-skill/data ./data/physiology-sport-skill/
COPY /home/hermes/.hermes/skills/books/issurin-training-skill/data ./data/issurin-training-skill/

# Create symlinks for the searcher to find data at expected paths
RUN mkdir -p /app/skills/physiology-sport-skill/data \
    && ln -sf /app/data/physiology-sport-skill /app/skills/physiology-sport-skill/data \
    && mkdir -p /app/skills/issurin-training-skill/data \
    && ln -sf /app/data/issurin-training-skill /app/skills/issurin-training-skill/data

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["python", "server.py"]