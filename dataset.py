import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
import json

# Безопасные аугментации для мультиспектральных данных (без CLAHE, сохраняющие каналы)
train_aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(scale=(0.95, 1.05), translate_percent=0.03, rotate=(-10, 10), p=0.5),
], p=1.0)

val_aug = A.Compose([])


class MarineDataset(Dataset):
    """Датасет для загрузки предобработанных патчей."""
    def __init__(self, npy_path, meta_path, augment=False):
        self.data = np.load(npy_path, allow_pickle=True)
        self.augment = augment
        with open(meta_path) as f:
            meta = json.load(f)
        mean = np.array(meta["mean"], dtype=np.float32)
        std = np.array(meta["std"], dtype=np.float32)
        std[std < 1e-6] = 1.0
        self.mean = mean[:, None, None]
        self.std = std[:, None, None]
        self.channels = mean.shape[0]

    def __len__(self):
        return len(self.data)

    def spectral_noise(self, img, std=0.005):
        """Мультипликативный гауссов шум, безопасный для спектральных каналов."""
        noise = np.random.normal(1.0, std, (self.channels, 1, 1)).astype(np.float32)
        return img * noise

    def __getitem__(self, idx):
        img, label = self.data[idx]
        img = img.astype(np.float32)
        img = np.nan_to_num(img)
        if self.augment:
            img = self.spectral_noise(img, std=0.005)
        # Переводим в HWC для albumentations
        img = np.transpose(img, (1, 2, 0))
        if self.augment:
            img = train_aug(image=img)["image"]
        img = np.transpose(img, (2, 0, 1))
        # Нормализация
        img = (img - self.mean) / self.std
        x = torch.from_numpy(img).float()
        y = torch.tensor(label, dtype=torch.float32)
        return x, y