# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variables (you can also set these in Heroku's dashboard)
ENV PORT=8000
ENV HOST=0.0.0.0
ENV MAX_PROMPT_LENGTH=4000
ENV STATIC_FILES_DIR="/app/frontend/dist"

# No CMD needed here, as heroku.yml defines the run command