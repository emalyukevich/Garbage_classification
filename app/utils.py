from torchvision import transforms
from PIL import Image
from io import BytesIO
import torch

# Простейшие трансформации, такие же как валидационные
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return val_transforms(image).unsqueeze(0)  # Добавляем батч-дименсию
