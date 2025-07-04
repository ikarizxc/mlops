# Use the official Python 3.12 base image (choose a variant like slim or alpine for smaller images)
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

ENV PYTHONPATH=/app

# Copy the requirements file into the container
COPY airflow/docker/data_preparation/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY airflow/scripts/data_preparation ./scripts/data_preparation
COPY airflow/scripts/preprocessors ./scripts/preprocessors