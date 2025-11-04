# Create & Activate Project Environment
#### Create the environment
```
conda env create -f environment.yml
```

##### Activate it
```
conda activate mura-env
```

# Data Preprocessing and Loader Setup

This project uses the MURA dataset to classify musculoskeletal X-rays as normal (0) or abnormal (1).

## Provided

- `MURADataset`: Custom PyTorch Dataset class
- `train_loader` and `val_loader` (see `data_preprocessing.ipynb`)
- Image transforms: `get_train_transforms()`, `get_val_transforms()`
- CSVs: `train_labeled_studies.csv`, `valid_labeled_studies.csv`
- Preview notebook: `notebooks/data_preprocessing.ipynb`
- Train/Val/Test Split: use `scripts/split_train_val.py` to split csvs

## Data Structure

- `data/raw/MURA-v1.1/`: Unzipped MURA dataset
- `data/splits/`: CSVs with study paths and binary labels
- Each study folder contains multiple `.png` images

## Splits

- Split is defined per study (not per image)
- Labels are at the study level, applied to all images in that folder


- The original `train_labeled_studies.csv` from MURA was split into:
  - `train_labeled_studies_split.csv` (80%) — used for training
  - `val_labeled_studies_split.csv` (20%) — used for validation
- The original `valid_labeled_studies.csv` is used as the **test set**
- Stratified split preserves class balance

## Usage

```python
from utils.mura_dataset import MURADataset
from utils.transforms import get_train_transforms, get_val_transforms
from torch.utils.data import DataLoader

train_dataset = MURADataset(
    csv_file="data/splits/train_labeled_studies_split.csv", #ensure correct split csv
    transform=get_train_transforms(),
    root_dir="data/raw"
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
