"""
AdaptSign — Indian Dataset Loader
IKS: Lok Vigyan (Local Knowledge)

Loads Indian Traffic Sign dataset from folder structure:
Images/
 ├── 0/
 ├── 1/
 ├── 2/
 ...
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, datasets
from PIL import Image


# ─────────────────────────────────────────
# SPLIT FUNCTION
# ─────────────────────────────────────────

def _split_indices(n, test_split=0.2, seed=42):
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()

    test_size = int(n * test_split)
    test_idx = perm[:test_size]
    train_idx = perm[test_size:]

    return train_idx, test_idx


# ─────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────

def get_transforms(mode='train', img_size=64):

    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            ),
        ])


# ─────────────────────────────────────────
# WEATHER TASK WRAPPER
# ─────────────────────────────────────────

class WeatherTaskDataset(Dataset):

    def __init__(self, base_dataset, weather_task=0, img_size=64, mode='train'):
        self.base_dataset = base_dataset
        self.weather_task = weather_task

        from data.augment import WeatherAugmenter
        self.augmenter = WeatherAugmenter()

        # Use clean deterministic transforms for test — no random flips/rotations
        self.transform = get_transforms(mode, img_size)
        self.weather_names = ['Sunny', 'Rainy', 'Foggy', 'Night']

        print(f"[TASK] {self.weather_names[weather_task]} ({mode})")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # base_dataset now returns raw PIL images (ImageFolder has no transform)
        img, label = self.base_dataset[idx]

        # Safety: if somehow already a tensor, convert back
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)

        # Apply weather augmentation (on PIL image, before normalization)
        if self.weather_task == 1:
            img = self.augmenter.add_rain(img)
        elif self.weather_task == 2:
            img = self.augmenter.add_fog(img)
        elif self.weather_task == 3:
            img = self.augmenter.add_night(img)

        img = self.transform(img)

        return img, label


# ─────────────────────────────────────────
# MAIN FUNCTION (IMPORTANT)
# ─────────────────────────────────────────

def get_sequential_tasks(data_dir, batch_size=64, img_size=64):

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset not found: {data_dir}")

    # Load dataset with NO transform — return raw PIL images.
    # WeatherTaskDataset applies the correct transform (train or test) exactly once.
    full_dataset = datasets.ImageFolder(
        root=data_dir,
        transform=None          # <-- Bug 3 fix: no transform here
    )

    print(f"[DATA] Total images: {len(full_dataset)}")
    print(f"[DATA] Classes: {len(full_dataset.classes)}")

    # Split train/test
    train_idx, test_idx = _split_indices(len(full_dataset))

    train_base = Subset(full_dataset, train_idx)
    test_base = Subset(full_dataset, test_idx)

    tasks = []

    for i in range(4):
        # Bug 2 fix: pass mode so test dataset gets clean deterministic transforms
        train_ds = WeatherTaskDataset(train_base, i, img_size, mode='train')
        test_ds  = WeatherTaskDataset(test_base,  i, img_size, mode='test')

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            persistent_workers=True
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=True
        )

        tasks.append((train_loader, test_loader))

    return tasks