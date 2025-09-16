# FedZK Docker Deployment Example
# Copyright (c) 2025 Aaryan Guglani and FedZK Contributors
# SPDX-License-Identifier: MIT

# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch torchvision

# Install FedZK from the current directory
RUN pip install -e .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Create a non-root user and switch to it
RUN useradd -m fedzk
USER fedzk

# Define environment variable
ENV FEDZK_MODE="production"
ENV FEDZK_LOG_LEVEL="info"

# Run the command to start the coordinator server
CMD ["python", "-m", "fedzk.coordinator.server", "--host", "0.0.0.0", "--port", "8000"]

# Usage instructions:
# 1. Build the image: docker build -t fedzk:latest .
# 2. Run the coordinator: docker run -p 8000:8000 fedzk:latest
# 3. For client deployment: docker run fedzk:latest python -m fedzk.client train 