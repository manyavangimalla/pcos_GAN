"""
Helper functions for PCOS detection project.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import yaml
from torch.utils.data import Dataset, DataLoader
import cv2
from datetime import datetime

def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to config file
        
    Returns:
        Dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: Dict, save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config (Dict): Configuration dictionary
        save_path (str): Path to save config file
    """
    with open(save_path, 'w') as f:
        yaml.dump(config, f)

class PCOSDataset(Dataset):
    """Dataset class for PCOS data."""
    def __init__(self,
                 image_dir: str,
                 hormonal_data_path: str,
                 clinical_records_path: str,
                 transform=None):
        """
        Initialize dataset.
        
        Args:
            image_dir (str): Directory containing ultrasound images
            hormonal_data_path (str): Path to hormonal data CSV
            clinical_records_path (str): Path to clinical records file
            transform: Optional transform to apply to images
        """
        self.image_dir = image_dir
        self.transform = transform
        
        # Load hormonal data
        self.hormonal_data = np.load(hormonal_data_path)
        
        # Load clinical records
        with open(clinical_records_path, 'r') as f:
            self.clinical_records = f.readlines()
        
        # Get image paths
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ]
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict:
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        # Get hormonal data and clinical record
        hormonal_data = self.hormonal_data[idx]
        clinical_record = self.clinical_records[idx]
        
        return {
            'image': image,
            'hormonal_data': hormonal_data,
            'clinical_record': clinical_record
        }

def create_dataloader(dataset: Dataset,
                     batch_size: int,
                     shuffle: bool = True,
                     num_workers: int = 4) -> DataLoader:
    """
    Create DataLoader for dataset.
    
    Args:
        dataset (Dataset): Dataset to create loader for
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle data
        num_workers (int): Number of worker processes
        
    Returns:
        DataLoader: DataLoader for the dataset
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

def visualize_batch(batch: Dict,
                   num_images: int = 8,
                   save_path: Optional[str] = None):
    """
    Visualize a batch of images.
    
    Args:
        batch (Dict): Batch of data
        num_images (int): Number of images to display
        save_path (str): Optional path to save visualization
    """
    images = batch['image'][:num_images]
    
    # Create grid
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.ravel()
    
    for idx, img in enumerate(images):
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy().transpose(1, 2, 0)
        
        axes[idx].imshow(img)
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def save_model(model: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              epoch: int,
              loss: float,
              save_path: str):
    """
    Save model checkpoint.
    
    Args:
        model (nn.Module): Model to save
        optimizer (Optimizer): Optimizer to save
        epoch (int): Current epoch
        loss (float): Current loss
        save_path (str): Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    
    torch.save(checkpoint, save_path)

def load_model(model: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              checkpoint_path: str) -> Tuple[torch.nn.Module, torch.optim.Optimizer, int, float]:
    """
    Load model checkpoint.
    
    Args:
        model (nn.Module): Model to load into
        optimizer (Optimizer): Optimizer to load into
        checkpoint_path (str): Path to checkpoint
        
    Returns:
        Tuple: (model, optimizer, epoch, loss)
    """
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer, checkpoint['epoch'], checkpoint['loss']

def setup_experiment(experiment_name: str, base_dir: str = '../results') -> str:
    """
    Set up experiment directory with timestamp.
    
    Args:
        experiment_name (str): Name of the experiment
        base_dir (str): Base directory for results
        
    Returns:
        str: Path to experiment directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'visualizations'), exist_ok=True)
    
    return experiment_dir

def main():
    """Example usage of helper functions."""
    # Example configuration
    config = {
        'experiment_name': 'pcos_detection',
        'data_dir': '../data',
        'batch_size': 32,
        'num_workers': 4
    }
    
    # Set up experiment
    experiment_dir = setup_experiment(config['experiment_name'])
    
    # Save config
    save_config(config, os.path.join(experiment_dir, 'config.yaml'))
    
    print(f"Experiment directory created at: {experiment_dir}")

if __name__ == "__main__":
    main() 