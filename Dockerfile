# Use an official Python runtime as a parent image
# using 3.10 which is stable for both torch and tensorflow as of late 2024
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# gcc and python3-dev are often needed for building python packages like shap
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Upgrade pip first to avoid issues with older pip versions
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the working directory contents into the container at /app
COPY . .

# Expose port 8501 for Streamlit
EXPOSE 8501

# Healthcheck to ensure the app is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Command to run the application
# server.address=0.0.0.0 is crucial for docker/cloud accessibility
ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
