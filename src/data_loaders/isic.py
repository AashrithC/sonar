import os
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, random_split
from PIL import Image
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any


class ISICDataset(Dataset):
    """
    ISIC (International Skin Imaging Collaboration) Dataset Class.
    This dataset contains dermoscopic images for skin lesion analysis.
    """
    def __init__(self, 
                 root_dir: str, 
                 split: str = 'train',
                 transform=None, 
                 target_transform=None,
                 download: bool = False) -> None:
        """
        Args:
            root_dir: Directory with the ISIC dataset.
            split: 'train' or 'test', split to use
            transform: Optional transform to be applied on images
            target_transform: Optional transform to be applied on labels
            download: If True, downloads the dataset from the internet
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # Create directory if it doesn't exist
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
            
        # Download dataset if requested and not already present
        if download and not self._check_exists():
            self._download()
            
        # Load dataset
        if not self._check_exists():
            raise RuntimeError('Dataset not found. Use download=True to download it')
            
        # Load metadata
        self.metadata = pd.read_csv(os.path.join(self.root_dir, 'ISIC_2019_Training_GroundTruth.csv'))
        
        # Get image paths and labels
        self.image_paths = []
        self.labels = []
        
        for idx, row in self.metadata.iterrows():
            image_id = row['image']
            image_path = os.path.join(self.root_dir, 'ISIC_2019_Training_Input', f'{image_id}.jpg')
            
            if os.path.exists(image_path):
                self.image_paths.append(image_path)
                
                # Convert diagnosis to numeric label (8 classes in ISIC 2019)
                label = 0  # Default to 'melanoma'
                for i, col in enumerate(self.metadata.columns[1:]):  # Skip 'image' column
                    if row[col] == 1.0:
                        label = i
                        break
                        
                self.labels.append(label)
        
        # Split into train and test
        if self.split == 'train':
            indices = list(range(len(self.image_paths)))
            split_idx = int(0.8 * len(indices))
            self.image_paths = [self.image_paths[i] for i in indices[:split_idx]]
            self.labels = [self.labels[i] for i in indices[:split_idx]]
        else:  # test
            indices = list(range(len(self.image_paths)))
            split_idx = int(0.8 * len(indices))
            self.image_paths = [self.image_paths[i] for i in indices[split_idx:]]
            self.labels = [self.labels[i] for i in indices[split_idx:]]
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label
    
    def _check_exists(self) -> bool:
        return os.path.exists(os.path.join(self.root_dir, 'ISIC_2019_Training_GroundTruth.csv')) and \
               os.path.exists(os.path.join(self.root_dir, 'ISIC_2019_Training_Input'))
    
    def _download(self) -> None:
        """
        Download the ISIC dataset if it doesn't exist already.
        Note: Due to the large size of the dataset, we provide instructions for manual download.
        """
        print("The ISIC 2019 dataset needs to be downloaded manually due to its large size.")
        print("Please download the dataset from https://challenge.isic-archive.com/data/")
        print("and place it in the following directory structure:")
        print(f"{self.root_dir}/ISIC_2019_Training_Input/ - containing all .jpg images")
        print(f"{self.root_dir}/ISIC_2019_Training_GroundTruth.csv - containing the labels")
        raise RuntimeError("Dataset not found. Please download it manually.")


class ISICDatasetWrapper:
    """
    Wrapper class for ISIC Dataset to match the interface expected by SONAR.
    """
    def __init__(self, dpath: str) -> None:
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.num_channels = 3
        self.image_size = 224
        self.num_cls = 8  # ISIC 2019 has 8 classes
        
        # Define transformations
        train_transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(20),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            T.ToTensor(),
            T.Normalize(self.mean, self.std)
        ])
        
        test_transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(self.mean, self.std)
        ])
        
        # Create datasets
        self.train_dset = ISICDataset(
            root_dir=dpath,
            split='train',
            transform=train_transform,
            download=False
        )
        
        self.test_dset = ISICDataset(
            root_dir=dpath,
            split='test',
            transform=test_transform,
            download=False
        ) 