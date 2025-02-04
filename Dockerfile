# Stage 1: Build frontend
FROM node:18-slim AS frontend-builder

WORKDIR /app

# Copy package files for frontend
COPY package*.json ./
COPY frontend/package*.json ./frontend/

# Install dependencies
RUN npm install
RUN cd frontend && npm install

# Copy frontend source
COPY frontend/ ./frontend/

# Build frontend
RUN cd frontend && npm run build

# Stage 2: Python application
FROM python:3.11.5-slim

WORKDIR /app

# Copy frontend build from previous stage
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Create static directory and copy frontend build
RUN mkdir -p static
RUN cp -r frontend/dist/* static/

# Copy Python requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PORT=8000
ENV HOST=0.0.0.0
ENV MAX_PROMPT_LENGTH=4000
ENV STATIC_FILES_DIR="/app/frontend/dist"

# Expose port
EXPOSE 8000

# Command to run the application
CMD gunicorn api.main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --timeout 120 --bind 0.0.0.0:$PORT