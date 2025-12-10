import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class DriverBehaviorDataset(Dataset):
    """Dataset with person-based split to prevent data leakage"""
    
    # Class names mapping
    CLASS_NAMES = {
        0: 'Safe driving',
        1: 'Texting - right hand',
        2: 'Talking on the phone - right hand',
        3: 'Texting - left hand',
        4: 'Talking on the phone - left hand',
        5: 'Operating the radio',
        6: 'Drinking',
        7: 'Reaching behind',
        8: 'Tidying up hair or applying makeup',
        9: 'Talking to passenger'
    }
    
    CLASS_CODES = [f'c{i}' for i in range(10)]
    
    def __init__(self, root_dir, person_list, transform=None):
        """
        Args:
            root_dir: Path to HowDir folder
            person_list: List of person names to include in this split
            transform: Optional transform to apply to images
        """
        self.root_dir = Path(root_dir)
        self.person_list = person_list
        self.transform = transform
        
        # Build list of all image paths and labels
        self.samples = []
        self.class_names = self.CLASS_CODES
        
        for person in person_list:
            person_dir = self.root_dir / person
            for class_idx, class_name in enumerate(self.class_names):
                class_dir = person_dir / class_name
                if class_dir.exists():
                    for img_file in class_dir.glob('*.jpg'):
                        self.samples.append({
                            'path': img_file,
                            'label': class_idx,
                            'class_name': class_name,
                            'person': person
                        })
        
        print(f"Loaded {len(self.samples)} samples from {len(person_list)} people")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, sample['label']
    
    def get_sample_info(self, idx):
        """Get metadata for a sample"""
        return self.samples[idx]


def create_person_splits(all_persons, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    """
    Split persons into train/val/test sets
    
    Args:
        all_persons: List of all person names
        train_ratio, val_ratio, test_ratio: Split ratios (should sum to 1.0)
        seed: Random seed for reproducibility
    """
    import random
    random.seed(seed)
    
    persons = all_persons.copy()
    random.shuffle(persons)
    
    n_total = len(persons)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_persons = persons[:n_train]
    val_persons = persons[n_train:n_train + n_val]
    test_persons = persons[n_train + n_val:]
    
    print(f"Split Summary:")
    print(f"  Train: {len(train_persons)} people - {train_persons}")
    print(f"  Val:   {len(val_persons)} people - {val_persons}")
    print(f"  Test:  {len(test_persons)} people - {test_persons}")
    
    return train_persons, val_persons, test_persons


# Example usage
if __name__ == "__main__":
    # Define all persons in dataset
    all_persons = ['zakaria', 'hajar', 'sara', 'hamza', 'achraf', 
                   'mohammed', 'amal', 'youssef', 'mohssine']
    
    # Create person splits (adjust ratios as needed)
    train_persons, val_persons, test_persons = create_person_splits(
        all_persons, 
        train_ratio=0.56,  # 5 people
        val_ratio=0.22,    # 2 people  
        test_ratio=0.22,   # 2 people
        seed=42
    )
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Path to your dataset
    dataset_root = '/kaggle/input/howdir-safedriving-benchmarking-sample/HowDir'
    
    # Create datasets
    train_dataset = DriverBehaviorDataset(dataset_root, train_persons, transform)
    val_dataset = DriverBehaviorDataset(dataset_root, val_persons, transform)
    test_dataset = DriverBehaviorDataset(dataset_root, test_persons, transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")