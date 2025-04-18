from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch

from config import class_names, model_path, device
from model import create_model
from utils import preprocess_image

app = FastAPI(title="Garbage Classification API")

# Загружаем модель
model = create_model(num_classes=len(class_names), model_path=model_path, device=device)

@app.get("/")
def root():
    return {"message": "Добро пожаловать в API классификации мусора"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Чтение изображения
        image_bytes = await file.read()
        input_tensor = preprocess_image(image_bytes).to(device)

        # Предсказание
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted_class = torch.max(outputs, 1)
            predicted_label = class_names[predicted_class.item()]

        return {"predicted_class": predicted_label}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

