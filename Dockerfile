# Use official Python 3.12 image
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first (so Docker caches this layer)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose the port Gradio runs on
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
