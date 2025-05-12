# === ИМПОРТЫ ===
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from albumentations.pytorch import ToTensorV2
import albumentations as A

from utils import calculate_metrics
from early_stopping import EarlyStopping
from dataset import AlbumentationsTransform
from model import create_model
from tqdm import tqdm

# === АУГМЕНТАЦИИ ===
train_transform = AlbumentationsTransform(
    A.Compose([
        A.RandomResizedCrop((224, 224), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.MotionBlur(blur_limit=3, p=0.3),
        A.GaussNoise(p=0.2),
        A.CLAHE(clip_limit=2.0, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
)

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# === ГЛАВНАЯ ФУНКЦИЯ ОБУЧЕНИЯ ===
def main():
    # Пути к данным
    train_dir = "../data/train"
    val_dir = "../data/val"

    # Datasets и Dataloaders
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    num_classes = len(train_dataset.classes)

    # Модель
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes)
    model = model.to(device)

    # Оптимизация
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    # EarlyStopping
    early_stopping = EarlyStopping(patience=7, verbose=True, path='../models/best_model.pth')
    best_val_f1 = 0.0
    num_epochs = 50

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # === Обучение ===
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []

        for images, labels in tqdm(train_loader, desc="Training", leave=False):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_acc, train_prec, train_rec, train_f1 = calculate_metrics(train_labels, train_preds)

        # === Валидация ===
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation", leave=False):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc, val_prec, val_rec, val_f1 = calculate_metrics(val_labels, val_preds)

        # === Логирование ===
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Prec: {train_prec:.4f} | Rec: {train_rec:.4f} | F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f}")

        # === Сохранение лучшей модели ===
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            os.makedirs("../models", exist_ok=True)
            torch.save(model.state_dict(), "../models/best_model.pth")
            print("New best model saved!")

        # === EarlyStopping ===
        early_stopping(val_f1, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break


# === ЗАПУСК ТОЛЬКО ЕСЛИ ЗАПУСТИЛИ СКРИПТ ===
if __name__ == "__main__":
    main()
