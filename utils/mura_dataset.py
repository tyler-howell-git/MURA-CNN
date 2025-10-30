import os
import pandas as pd
from glob import glob
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MURADataset(Dataset):
    def __init__(self, csv_file, transform=None, root_dir=None):
        self.data = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for i in range(len(self.data)):
            study_path, label_raw = self.data.iloc[i]
            label = int(label_raw)  # Convert numeric label to Python int (0 or 1)

            study_full_path = os.path.join(self.root_dir, study_path)
            image_paths = glob(os.path.join(study_full_path, "*.png"))

            for img_path in image_paths:
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
