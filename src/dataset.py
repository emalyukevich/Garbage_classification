import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class GarbageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: path to the data folder (e.g. data/train)
        transform: torchvision transformations
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []              # [(image_path, label_index), ...]
        self.class_to_idx = {}         # {'glass': 0, 'metal': 1, ...}

        self._load_samples()

    def _load_samples(self):
        classes = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d)) and not d.startswith(".")
        ])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        self.classes = list(self.class_to_idx.keys())
        
        for cls in classes:
            cls_path = os.path.join(self.root_dir, cls)
            for fname in os.listdir(cls_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(cls_path, fname)
                    self.samples.append((full_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        return image, label

class AlbumentationsTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        image = np.array(img)
        augmented = self.transform(image=image)
        return augmented["image"]
