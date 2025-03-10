import os
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any, Union


class ChestXrayDataset(Dataset):
    """
    ChestX-ray14 Dataset Class.
    This dataset contains chest X-ray images with 14 disease labels.
    """
    def __init__(self, 
                 root_dir: str, 
                 split: str = 'train',
                 transform: Optional[Any] = None, 
                 target_transform: Optional[Any] = None,
                 download: bool = False) -> None:
        """
        Args:
            root_dir: Directory with the ChestX-ray14 dataset.
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
        self.metadata = pd.read_csv(os.path.join(self.root_dir, 'Data_Entry_2017.csv'))
        
        # Define the 14 disease classes
        self.classes = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
                        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 
                        'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        
        # Get image paths and labels
        self.image_paths = []
        self.labels = []
        
        for idx, row in self.metadata.iterrows():
            image_id = row['Image Index']
            image_path = os.path.join(self.root_dir, 'images', image_id)
            
            if os.path.exists(image_path):
                self.image_paths.append(image_path)
                
                # Parse finding labels
                finding_labels = row['Finding Labels'].split('|')
                
                # Convert to multi-hot encoding (multi-label classification)
                label = torch.zeros(len(self.classes))
                for finding in finding_labels:
                    if finding != 'No Finding' and finding in self.classes:
                        label_idx = self.classes.index(finding)
                        label[label_idx] = 1.0
                        
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
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label
    
    def _check_exists(self) -> bool:
        return os.path.exists(os.path.join(self.root_dir, 'Data_Entry_2017.csv')) and \
               os.path.exists(os.path.join(self.root_dir, 'images'))
    
    def _download(self) -> None:
        """
        Download the ChestX-ray14 dataset if it doesn't exist already.
        Note: Due to the large size of the dataset, we provide instructions for manual download.
        """
        print("The ChestX-ray14 dataset needs to be downloaded manually due to its large size.")
        print("Please download the dataset from https://nihcc.app.box.com/v/ChestXray-NIHCC")
        print("and place it in the following directory structure:")
        print(f"{self.root_dir}/images/ - containing all X-ray images")
        print(f"{self.root_dir}/Data_Entry_2017.csv - containing the labels")
        raise RuntimeError("Dataset not found. Please download it manually.")


class ChestXrayDatasetWrapper:
    """
    Wrapper class for ChestX-ray14 Dataset to match the interface expected by SONAR.
    """
    def __init__(self, dpath: str) -> None:
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.num_channels = 3
        self.image_size = 224
        self.num_cls = 14  # ChestX-ray14 has 14 disease classes
        
        # Define transformations
        train_transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(self.mean, self.std)
        ])
        
        test_transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(self.mean, self.std)
        ])
        
        # Create datasets
        self.train_dset = ChestXrayDataset(
            root_dir=dpath,
            split='train',
            transform=train_transform,
            download=False
        )
        
        self.test_dset = ChestXrayDataset(
            root_dir=dpath,
            split='test',
            transform=test_transform,
            download=False
        ) 