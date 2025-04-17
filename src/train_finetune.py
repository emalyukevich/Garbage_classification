import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from utils import calculate_metrics
from early_stopping import EarlyStopping
from model import create_model
from tqdm import tqdm

# Пути к данным и модели
train_dir = "../data/train"
val_dir = "../data/val"
checkpoint_path = "../models/best_model.pth"

# Упрощённые аугментации (fine-tuning)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Datasets
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

num_classes = len(train_dataset.classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Инициализация модели и загрузка весов
model = create_model(num_classes)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model = model.to(device)

# Размораживаем все слои для тонкой настройки (если требуется)
for param in model.parameters():
    param.requires_grad = True

# Оптимизатор и функция потерь
optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-3)
criterion = nn.CrossEntropyLoss()

# Fine-tuning: обучаем 15 эпох
num_epochs = 15
best_val_f1 = 0.0
early_stopping = EarlyStopping(patience=5, verbose=True, path="../models/best_model_finetuned.pth")

for epoch in range(num_epochs):
    print(f"\nFine-Tune Epoch {epoch+1}/{num_epochs}")

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
        torch.save(model.state_dict(), "../models/best_model_finetuned.pth")
        print("New best fine-tuned model saved!")

    # === Early Stopping ===
    early_stopping(val_f1, model)
    if early_stopping.early_stop:
        print("Early stopping triggered.")
        break
