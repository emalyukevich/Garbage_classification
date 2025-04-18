import torch
import os

from torchvision import datasets, transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # app/
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))  # Garbage_classification/

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder(os.path.join(ROOT_DIR, "data/val"), transform=val_transform)

model_path = os.path.join(ROOT_DIR, "models/best_model_finetuned.pth")
class_names = val_dataset.classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

