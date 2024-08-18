# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    FLASK_ENV=production \
    UPLOAD_FOLDER=uploads \
    ALLOWED_EXTENSIONS="png,jpg,jpeg" \
    PORT=10000

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Create the uploads directory
RUN mkdir -p /app/uploads

# Expose the port on which the Flask app will run
EXPOSE 10000

# Run the Flask application with Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000"]
