from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.responses import FileResponse
from fastapi import Request

import torch
import os

from app.config import class_names, model_path, device
from app.model import create_model
from app.utils import preprocess_image

app = FastAPI(
    title="🧠 Garbage Classification API",
    description="""
📷 Классификация изображений мусора с помощью модели **EfficientNet-B0**, обученной на основе трансферного обучения.

### Возможности:
- 🚀 Предсказание класса мусора по изображению
- 🧪 Использование FastAPI для продакшен-инференса
- 📊 Аугментации, метрики, ранняя остановка, кастомный пайплайн

Разработано с прицелом на **ML-портфолио** и демонстрацию навыков **MLOps + CV + API**.
""",
    version="1.0.0"
)

# Загружаем модель
model = create_model(num_classes=len(class_names), model_path=model_path, device=device)

@app.get("/", summary="Главная страница", description="Приветственная страница API")
def read_root():
    return {"message": "Добро пожаловать в Garbage Classification API 🚀"}

@app.post(
    "/predict",
    summary="🔍 Предсказание класса по изображению",
    description="""
Загрузите изображение (формат JPEG/PNG), и модель вернет наиболее вероятный класс объекта.

⚠️ Ожидается изображение с цветовым каналом (RGB), размер автоматически масштабируется до 224x224.

**Пример запроса**:
- Тип: `multipart/form-data`
- Поле: `file`

**Пример кода (Python)**:
```python
import requests

url = "http://127.0.0.1:8000/predict"
with open("image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)
print(response.json())"""
    )
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

app.mount("/examples", StaticFiles(directory="app/examples"), name="examples")

@app.get(
    "/view_examples",
    summary="🖼️ Просмотр сгенерированных примеров",
    description="""
Показывает изображения, обработанные моделью, с предсказанными классами.

📂 Изображения берутся из папки `/examples`.

🧠 Предсказания формируются моделью классификации мусора.

---

### 🔗 Как посмотреть примеры:
✅ После генерации примеров через `/generate_examples`, нажми [сюда](http://localhost:8000/view_examples), чтобы увидеть результаты.

""",
    response_description="HTML-страница с примерами предсказаний"
)
async def view_examples(request: Request):
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    examples_dir = os.path.join(CURRENT_DIR, "examples")

    try:
        files = [f for f in os.listdir(examples_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    except FileNotFoundError:
        return JSONResponse(content={"error": "Папка examples не найдена"}, status_code=404)


    if not files:
        message = {"message": "Нет изображений для отображения."}
        if request.headers.get("accept") == "application/json":
            return JSONResponse(content=message, status_code=200)
        return HTMLResponse(content=f"<h3>{message['message']}</h3>", status_code=200)

    if request.headers.get("accept") == "application/json":
        return JSONResponse(
            content={"message": "Изображения успешно сгенерированы. Перейдите на http://localhost:8000/view_examples для просмотра."},
            status_code=200
        )

    # HTML визуализация
    html_content = """
    <html>
    <head>
        <title>📸 Примеры классификации</title>
        <style>
            body {
                font-family: 'Segoe UI', sans-serif;
                padding: 30px;
                background-color: #fafafa;
            }
            h2 {
                text-align: center;
                color: #333;
            }
            .gallery {
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 20px;
            }
            .card {
                width: 240px;
                background: white;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                overflow: hidden;
                text-align: center;
                transition: 0.3s;
            }
            .card:hover {
                transform: scale(1.02);
                box-shadow: 0 6px 16px rgba(0,0,0,0.15);
            }
            .label {
                font-size: 18px;
                font-weight: bold;
                background-color: #4caf50;
                color: white;
                padding: 10px;
                border-bottom: 1px solid #ddd;
            }
            img {
                width: 100%;
                height: 224px;
                object-fit: cover;
            }
        </style>
    </head>
    <body>
        <h2>🧠 Результаты предсказания модели</h2>
        <div class="gallery">
    """

    for file in files:
        image_url = f"/examples/{file}"
        predicted_class = os.path.splitext(file)[0].split("_")[-1].capitalize()
        html_content += f"""
        <div class="card">
            <div class="label">Predicted: {predicted_class}</div>
            <img src="{image_url}" alt="{predicted_class}">
        </div>
        """

    html_content += """
        </div>
    </body>
    </html>
    """

    return HTMLResponse(content=html_content, status_code=200)











    
    
