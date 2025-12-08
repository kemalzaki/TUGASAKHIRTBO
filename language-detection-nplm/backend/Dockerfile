FROM python:3.11-slim

WORKDIR /app

# Install system deps (minimal) and pip packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*

# Copy requirements from project root
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files into the image
COPY . /app

WORKDIR /app/backend

EXPOSE 5000

# Run the Flask app (for production prefer waitress)
CMD ["python", "app.py"]
