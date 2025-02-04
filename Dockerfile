# Use Python 3.11.5 slim base image
FROM python:3.11.5-slim

# Set working directory
WORKDIR /app

# Install Node.js and npm
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY package*.json ./
COPY frontend/package*.json ./frontend/

# Install dependencies
RUN npm install
RUN cd frontend && npm install

# Copy the rest of the application
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Build frontend
RUN cd frontend && npm run build

# Set environment variables
ENV PORT=8000
ENV HOST=0.0.0.0

# Expose port
EXPOSE 8000

# Command to run the application
CMD gunicorn api.main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --timeout 120 --bind 0.0.0.0:$PORT