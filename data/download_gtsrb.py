"""
AdaptSign — Data Loader
IKS Principle: Lok Vigyan (Local Knowledge)
Downloads GTSRB + prepares Indian sign extensions
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, datasets
from PIL import Image
import urllib.request
import zipfile


def _split_indices(n, test_split=0.2, seed=42):
    if n <= 0:
        return [], []
    test_size = int(round(n * float(test_split)))
    test_size = max(1, test_size)
    test_size = min(n - 1, test_size)

    g = torch.Generator()
    g.manual_seed(int(seed))
    perm = torch.randperm(n, generator=g).tolist()
    test_idx = perm[:test_size]
    train_idx = perm[test_size:]
    return train_idx, test_idx

# ─────────────────────────────────────────
# CLASS NAMES
# ─────────────────────────────────────────

GTSRB_CLASSES = [
    'Speed limit 20', 'Speed limit 30', 'Speed limit 50', 'Speed limit 60',
    'Speed limit 70', 'Speed limit 80', 'End speed limit 80', 'Speed limit 100',
    'Speed limit 120', 'No passing', 'No passing (trucks)', 'Priority road',
    'Priority at junction', 'Give way', 'Stop', 'No vehicles',
    'No trucks', 'No entry', 'Caution', 'Dangerous curve left',
    'Dangerous curve right', 'Double curve', 'Bumpy road', 'Slippery road',
    'Road narrows right', 'Road works', 'Traffic signals', 'Pedestrians',
    'Children crossing', 'Cyclists', 'Ice/snow', 'Wild animals',
    'End restrictions', 'Turn right ahead', 'Turn left ahead', 'Go ahead',
    'Go ahead or right', 'Go ahead or left', 'Keep right', 'Keep left',
    'Roundabout', 'End no passing', 'End no passing (trucks)'
]

# Indian-specific signs added (Lok Vigyan — local knowledge)
INDIAN_SIGN_CLASSES = [
    'Cattle Crossing',
    'Speed Breaker Ahead',
    'School Zone',
    'Ghat Road',
    'Accident Prone Zone',
    'Narrow Road Ahead',
    'Ferry Crossing',
    'Bullock Cart Prohibited'
]

ALL_CLASSES = GTSRB_CLASSES + INDIAN_SIGN_CLASSES
NUM_CLASSES = len(ALL_CLASSES)  # 43 + 8 = 51


# ─────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────

def get_transforms(mode='train', img_size=64):
    """
    IKS: Panchang — prepare for all seasons/conditions
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.1),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.3337, 0.3064, 0.3171],
                std=[0.2672, 0.2564, 0.2629]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.3337, 0.3064, 0.3171],
                std=[0.2672, 0.2564, 0.2629]
            )
        ])


# ─────────────────────────────────────────
# GTSRB DATASET
# ─────────────────────────────────────────

def get_gtsrb_loaders(data_dir='./data/gtsrb', batch_size=64, img_size=64):
    """
    Returns train/test DataLoaders for GTSRB.
    Downloads automatically if not present.
    """
    os.makedirs(data_dir, exist_ok=True)

    train_dataset = datasets.GTSRB(
        root=data_dir,
        split='train',
        transform=get_transforms('train', img_size),
        download=True
    )

    test_dataset = datasets.GTSRB(
        root=data_dir,
        split='test',
        transform=get_transforms('test', img_size),
        download=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=2, pin_memory=True
    )

    print(f"[DATA] Train: {len(train_dataset)} | Test: {len(test_dataset)}")
    return train_loader, test_loader


# ─────────────────────────────────────────
# SEQUENTIAL WEATHER TASK DATASET
# ─────────────────────────────────────────

class WeatherTaskDataset(Dataset):
    """
    IKS: Panchang — Sequential weather tasks
    Wraps GTSRB images and applies weather augmentation per task.

    Tasks:
        0 = Sunny   (original)
        1 = Rainy
        2 = Foggy
        3 = Night
    """

    def __init__(self, base_dataset, weather_task=0, img_size=64):
        self.base_dataset = base_dataset
        self.weather_task = weather_task
        self.img_size = img_size
        self.base_transform = get_transforms('test', img_size)

        # Import here to avoid circular import
        from data.augment import WeatherAugmenter
        self.augmenter = WeatherAugmenter()

        self.weather_names = ['Sunny', 'Rainy', 'Foggy', 'Night']
        print(f"[TASK] Weather Task {weather_task}: {self.weather_names[weather_task]}")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]

        # Convert tensor back to PIL for augmentation
        if isinstance(img, torch.Tensor):
            img_np = img.permute(1, 2, 0).numpy()
            img_np = (img_np * np.array([0.2672, 0.2564, 0.2629]) +
                      np.array([0.3337, 0.3064, 0.3171]))
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
        else:
            img_pil = img

        # Apply weather augmentation
        if self.weather_task == 1:
            img_pil = self.augmenter.add_rain(img_pil)
        elif self.weather_task == 2:
            img_pil = self.augmenter.add_fog(img_pil)
        elif self.weather_task == 3:
            img_pil = self.augmenter.add_night(img_pil)

        # Re-apply standard transform
        img_tensor = self.base_transform(img_pil)
        return img_tensor, label


def get_sequential_tasks(data_dir='./data/gtsrb', batch_size=64, img_size=64):
    """
    Returns list of (train_loader, test_loader) for each weather task.
    This is the core data pipeline for lifelong learning.

    Returns:
        tasks: list of 4 tuples, each (train_loader, test_loader)
    """
    # Base dataset
    train_base = datasets.GTSRB(
        root=data_dir, split='train',
        transform=get_transforms('train', img_size),
        download=True
    )
    test_base = datasets.GTSRB(
        root=data_dir, split='test',
        transform=get_transforms('test', img_size),
        download=True
    )

    tasks = []
    for weather_id in range(4):
        train_ds = WeatherTaskDataset(train_base, weather_task=weather_id, img_size=img_size)
        test_ds = WeatherTaskDataset(test_base, weather_task=weather_id, img_size=img_size)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            shuffle=True, num_workers=0
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size,
            shuffle=False, num_workers=0
        )
        tasks.append((train_loader, test_loader))

    return tasks


def get_indian_traffic_sign_tasks(
    root_dir='./data/Indian-Traffic Sign-Dataset/Images',
    batch_size=64,
    img_size=64,
    test_split=0.2,
    seed=42,
):
    """Create 4 sequential weather tasks from the Indian-Traffic Sign-Dataset.

    Expected structure:
        root_dir/
          0/...
          1/...
          2/...

    Returns:
        tasks: list[(train_loader, test_loader)] for 4 weather tasks
        class_names: list[str] in the order used for labels
    """
    if not os.path.exists(root_dir):
        raise FileNotFoundError(
            f"Indian dataset folder not found: {root_dir} (expected Images/<classid>/...)"
        )

    train_base_full = datasets.ImageFolder(
        root=root_dir,
        transform=get_transforms('train', img_size),
    )
    test_base_full = datasets.ImageFolder(
        root=root_dir,
        transform=get_transforms('test', img_size),
    )

    train_idx, test_idx = _split_indices(
        n=len(train_base_full),
        test_split=test_split,
        seed=seed,
    )

    train_base = Subset(train_base_full, train_idx)
    test_base = Subset(test_base_full, test_idx)

    class_names = [c for c, _ in sorted(train_base_full.class_to_idx.items(), key=lambda x: x[1])]

    tasks = []
    for weather_id in range(4):
        train_ds = WeatherTaskDataset(train_base, weather_task=weather_id, img_size=img_size)
        test_ds = WeatherTaskDataset(test_base, weather_task=weather_id, img_size=img_size)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        tasks.append((train_loader, test_loader))

    print(f"[INDIA] Loaded Indian-Traffic Sign-Dataset")
    print(f"[INDIA] Root: {root_dir}")
    print(f"[INDIA] Classes: {len(class_names)}")
    print(f"[INDIA] Train: {len(train_base)} | Test: {len(test_base)} (split={test_split})")
    return tasks, class_names


# ─────────────────────────────────────────
# INDIAN SIGN DATASET (Custom)
# ─────────────────────────────────────────

class IndianSignDataset(Dataset):
    """
    IKS: Lok Vigyan — Local Indian traffic sign knowledge
    Loads images from ./data/indian_signs/<class_name>/<img>.jpg

    Create folders manually and add ~50 photos each via phone camera
    or web search for Indian traffic sign images.
    """

    def __init__(self, root='./data/indian_signs', transform=None):
        self.root = root
        self.transform = transform or get_transforms('train')
        self.samples = []
        self.class_names = []

        if not os.path.exists(root):
            print(f"[WARN] Indian signs dir not found: {root}")
            print("[WARN] Create it and add sign folders with images.")
            return

        classes = sorted(os.listdir(root))
        self.class_names = classes

        for class_idx, class_name in enumerate(classes):
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((
                        os.path.join(class_dir, img_file),
                        class_idx + 43  # offset after GTSRB's 43 classes
                    ))

        print(f"[INDIAN] Loaded {len(self.samples)} images "
              f"from {len(classes)} Indian sign classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────

if __name__ == '__main__':
    print("Testing data pipeline...")
    train_loader, test_loader = get_gtsrb_loaders(batch_size=32)
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape} | Labels: {labels[:5]}")
    print("Data pipeline OK ✓")
