# Use a slim version of the official Python image as a base image
FROM python:3.11.9-slim

# Set the working directory in the container
WORKDIR /app

# Install Torch first because it's required but slowed to install
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu && \
    apt-get remove -y build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to leverage Docker layer caching
COPY requirements.txt /app/

RUN pip install -r requirements.txt && \
    apt-get remove -y build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Remove as no need after installation
RUN rm -f requirements.txt
RUN apt-get update && apt-get install -y curl
# Copy the rest of the application files
COPY service.py /app/model_server/service.py
COPY src/ /app/src
ENV PYTHONPATH=/app

EXPOSE 3000

WORKDIR /app/model_server

