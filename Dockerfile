# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY FUJI_DETECTOR/ ./FUJI_DETECTOR/

# Expose port
EXPOSE 5000

# Run the API server
CMD ["python", "app.py"]
