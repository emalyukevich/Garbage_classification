import torch
from torch import nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

class EfficientNetClassifier(nn.Module):
    """
    Классификатор на базе EfficientNet-B0 для задачи классификации изображений.

    Аргументы:
        num_classes (int): Количество целевых классов.

    Атрибуты:
        backbone (nn.Module): Предобученная модель EfficientNet-B0 с заменённым классификатором.
    """
    def __init__(self, num_classes: int):
        super(EfficientNetClassifier, self).__init__()

        # Загружаем предобученную модель EfficientNet-B0 с последними весами
        weights = EfficientNet_B0_Weights.DEFAULT
        self.backbone = models.efficientnet_b0(weights=weights)

        # Замораживаем сверточные слои (features) — Transfer Learning
        for param in self.backbone.features.parameters():
            param.requires_grad = False

        # Заменяем классификационный "head" на кастомный для num_classes
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямое распространение (Forward pass) входного батча через сеть.

        Аргументы:
            x (Tensor): Входной батч изображений формы (B, 3, 224, 224)

        Возвращает:
            Tensor: Предсказания формы (B, num_classes)
        """
        return self.backbone(x)


def create_model(num_classes: int, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> nn.Module:
    """
    Создание и отправка модели на нужное устройство (CPU или GPU).

    Аргументы:
        num_classes (int): Количество классов.
        device (str): Устройство ('cuda' или 'cpu').

    Возвращает:
        nn.Module: Инициализированная модель.
    """
    model = EfficientNetClassifier(num_classes)
    return model.to(device)


# Пример тестирования модели
if __name__ == "__main__":
    model = create_model(num_classes=6)
    x = torch.randn((2, 3, 224, 224))
    out = model(x)
    print(out.shape)  # -> torch.Size([2, 6])

