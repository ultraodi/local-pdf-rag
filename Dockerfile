FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so Docker caches this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY ui.py .

# Defaults; docker-compose overrides these
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434
ENV CHROMA_DIR=/data/chroma

EXPOSE 7860

CMD ["python", "ui.py"]