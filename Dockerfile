FROM python:3.8

# Set the working directory inside the container
WORKDIR /app

# Copy the model file from the model directory in your project to the desired location in the container
COPY model/cnn_deepfake_model.keras /home/app/model/cnn_deepfake_model.keras

# Copy the requirements.txt file and install the dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the templates (HTML files) to the container
COPY public/templates/ /app/templates/

# Copy static files (like images, CSS, JS) to the container
COPY static/ /app/static/

# Copy the rest of your application code to the container
COPY . .

# Expose the port your application will run on
EXPOSE 5100

# Command to run your application
CMD ["python", "app.py"]
