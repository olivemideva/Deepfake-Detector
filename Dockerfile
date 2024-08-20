FROM python:3.8

# Copy the model file from the model directory in your project to the desired location in the container
COPY model/cnn_deepfake_model.keras /home/app/model/cnn_deepfake_model.keras

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5100

CMD ["python", "app.py"]
