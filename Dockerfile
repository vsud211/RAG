# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install Poppler, curl, Tesseract, and additional system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and install Ollama
RUN curl -fsSL https://ollama.com/install.sh | bash

# Start the Ollama service and wait for it to be ready
RUN ollama serve & \
    sleep 10 && \
    ollama pull llama2

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Expose the Flask port
EXPOSE 5000

# Expose the port for the Ollama service
EXPOSE 11434

# Start the Ollama service and run the application
CMD ollama serve & python main.py
