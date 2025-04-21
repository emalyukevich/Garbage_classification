import os
import random
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

VAL_DIR = "../data/val"
SAVE_DIR = "examples"
PREDICT_URL = "http://127.0.0.1:8000/predict" 
NUM_IMAGES = 3  

os.makedirs(SAVE_DIR, exist_ok=True)

def get_random_images(val_dir, num_images):
    all_images = []
    for class_dir in os.listdir(val_dir):
        class_path = os.path.join(val_dir, class_dir)
        if os.path.isdir(class_path):
            images = [os.path.join(class_path, img) for img in os.listdir(class_path)]
            all_images.extend(images)
    return random.sample(all_images, num_images)

def predict_image(image_path):
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
        response = requests.post(PREDICT_URL, files=files)
        if response.status_code == 200:
            return response.json()["predicted_class"]
        else:
            print(f"Error with {image_path}: {response.text}")
            return "Ошибка"

def overlay_prediction(image_path, prediction, save_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", size=24)
    except:
        font = ImageFont.load_default()

    draw.text((10, 10), f"Prediction: {prediction}", font=font, fill=(255, 0, 0))
    image.save(save_path)

if __name__ == "__main__":
    images = get_random_images(VAL_DIR, NUM_IMAGES)
    for img_path in images:
        pred = predict_image(img_path)
        filename = os.path.basename(img_path)
        save_path = os.path.join(SAVE_DIR, f"pred_{filename}")
        overlay_prediction(img_path, pred, save_path)
        print(f"[✓] Saved: {save_path} — Prediction: {pred}")
