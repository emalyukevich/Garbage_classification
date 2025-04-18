import torch
from torch import nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super(EfficientNetClassifier, self).__init__()
        weights = EfficientNet_B0_Weights.DEFAULT
        self.backbone = models.efficientnet_b0(weights=weights)

        for param in self.backbone.features.parameters():
            param.requires_grad = False

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def create_model(num_classes: int, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> nn.Module:
    model = EfficientNetClassifier(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
