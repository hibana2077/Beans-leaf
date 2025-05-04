import os
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm


class BeansDataset(Dataset):
    def __init__(self, split="train", transform=None):
        """
        Args:
            split (str): Which dataset split to use ('train', 'validation', or 'test')
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataset = load_dataset("AI-Lab-Makerere/beans")[split]
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        
        # Convert to binary classification (0: healthy, 1: diseased)
        # Original labels: 0 (angular_leaf_spot), 1 (bean_rust), 2 (healthy)
        # Convert to: 0 (healthy), 1 (diseased)
        original_label = item['labels']
        binary_label = 0 if original_label == 2 else 1
        
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'labels': binary_label
        }


def get_data_loaders(batch_size=32, num_workers=4):
    """
    Create data loaders for training, validation and testing
    
    Args:
        batch_size (int): Batch size for the data loaders
        num_workers (int): Number of workers for data loading
        
    Returns:
        dict: Dictionary containing train, val, and test data loaders
    """
    # Define transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create datasets
    train_dataset = BeansDataset(split="train", transform=train_transform)
    val_dataset = BeansDataset(split="validation", transform=val_test_transform)
    test_dataset = BeansDataset(split="test", transform=val_test_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


if __name__ == "__main__":
    # Test the dataset
    dataset = BeansDataset(split="train")
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Sample: {sample}")
    
    # Test data loaders
    loaders = get_data_loaders(batch_size=4)
    
    for split, loader in loaders.items():
        print(f"{split} loader size: {len(loader)}")
    
    # Display a batch from the training loader
    for batch in tqdm(loaders['train'], desc="Testing training loader"):
        print(f"Batch shape: {batch['image'].shape}")
        print(f"Labels: {batch['labels']}")
        break