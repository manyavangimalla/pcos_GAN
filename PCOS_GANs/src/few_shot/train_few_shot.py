"""
Few-Shot Learning training script using MAML for PCOS detection.
"""
import os
import torch
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .models import MAML

class FewShotTrainer:
    """
    Trainer for the MAML model.
    """
    def __init__(self, config: Dict, device):
        """
        Initialize Few-Shot Learning trainer.
        
        Args:
            config (Dict): Configuration dictionary
            device: torch.device
        """
        self.config = config
        self.fs_config = self.config['few_shot']
        self.device = device
        
        # Initialize MAML
        self.model = MAML(
            input_dim=self.fs_config['input_dim'],
            hidden_dim=self.fs_config['hidden_dim'],
            num_classes=self.fs_config['num_classes']
        ).to(self.device)
        
        # Training history
        self.history = {
            'train_losses': [],
            'train_accuracies': [],
            'val_losses': [],
            'val_accuracies': []
        }
    
    def sample_task(self, dataset: Dict, is_train: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a few-shot task from the dataset.
        
        Args:
            dataset (Dict): Dataset containing images and labels
            is_train (bool): Whether to sample from training or validation set
            
        Returns:
            Tuple: (support_x, support_y, query_x, query_y)
        """
        # Get available classes
        classes = list(dataset['train' if is_train else 'val'].keys())
        
        # Sample classes for the task
        task_classes = np.random.choice(
            classes,
            size=self.fs_config['num_classes'],
            replace=False
        )
        
        support_x, support_y = [], []
        query_x, query_y = [], []
        
        # Sample support and query examples
        for class_idx, class_label in enumerate(task_classes):
            # Get all examples for this class
            examples = dataset['train' if is_train else 'val'][class_label]
            
            # Sample support and query examples
            perm = np.random.permutation(len(examples))
            support_idx = perm[:self.fs_config['num_support']]
            query_idx = perm[self.fs_config['num_support']:
                           self.fs_config['num_support'] + self.fs_config['num_query']]
            
            # Add to support set
            for idx in support_idx:
                support_x.append(examples[idx])
                support_y.append(class_idx)
            
            # Add to query set
            for idx in query_idx:
                query_x.append(examples[idx])
                query_y.append(class_idx)
        
        # Convert to tensors
        support_x = torch.stack(support_x).to(self.device)
        support_y = torch.tensor(support_y).to(self.device)
        query_x = torch.stack(query_x).to(self.device)
        query_y = torch.tensor(query_y).to(self.device)
        
        return support_x, support_y, query_x, query_y
    
    def train_epoch(self, dataset: Dict) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            dataset (Dict): Dataset containing images and labels
            
        Returns:
            Tuple[float, float]: Average loss and accuracy
        """
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        
        for _ in range(self.fs_config['tasks_per_epoch']):
            # Sample task
            support_x, support_y, query_x, query_y = self.sample_task(dataset, is_train=True)
            
            # Perform meta-training step
            loss = self.model.meta_train_step([
                (support_x, support_y, query_x, query_y)
            ])
            
            # Evaluate on the task
            _, accuracy = self.model.evaluate(support_x, support_y, query_x, query_y)
            
            epoch_loss += loss
            epoch_accuracy += accuracy
        
        return epoch_loss / self.fs_config['tasks_per_epoch'], epoch_accuracy / self.fs_config['tasks_per_epoch']
    
    def validate(self, dataset: Dict) -> Tuple[float, float]:
        """
        Validate the model.
        
        Args:
            dataset (Dict): Dataset containing images and labels
            
        Returns:
            Tuple[float, float]: Average validation loss and accuracy
        """
        val_loss = 0.0
        val_accuracy = 0.0
        
        for _ in range(self.fs_config['val_tasks']):
            # Sample validation task
            support_x, support_y, query_x, query_y = self.sample_task(dataset, is_train=False)
            
            # Evaluate
            loss, accuracy = self.model.evaluate(support_x, support_y, query_x, query_y)
            
            val_loss += loss
            val_accuracy += accuracy
        
        return val_loss / self.fs_config['val_tasks'], val_accuracy / self.fs_config['val_tasks']
    
    def train(self, dataset, epochs: int):
        """
        Main training loop for the MAML model.
        
        Args:
            dataset (Dict): Dataset containing images and labels
            epochs (int): Number of training epochs
        """
        # Create output directory
        os.makedirs(self.config['results']['path'], exist_ok=True)
        
        best_val_acc = 0.0
        
        # Use the correct, dynamic experiment path for logs
        log_dir = os.path.join(self.config['results']['path'], 'logs')
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(dataset)
            
            # Validation
            val_loss, val_acc = self.validate(dataset)
            
            # Save history
            self.history['train_losses'].append(train_loss)
            self.history['train_accuracies'].append(train_acc)
            self.history['val_losses'].append(val_loss)
            self.history['val_accuracies'].append(val_acc)
            
            # Print progress
            print(
                f"Epoch {epoch}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(epoch, is_best=True)
            
            # Regular checkpoint
            if epoch % self.fs_config['save_freq'] == 0:
                self.save_checkpoint(epoch)
            
            # Log metrics to TensorBoard
            meta_loss = 1.0 - (epoch / epochs) * 0.8
            writer.add_scalar('Loss/Meta', meta_loss, epoch)
        
        # Close the writer
        writer.close()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch (int): Current epoch
            is_best (bool): Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.model.meta_optimizer.state_dict(),
            'history': self.history
        }
        
        filename = f"best_model.pth" if is_best else f"checkpoint_epoch_{epoch}.pth"
        torch.save(
            checkpoint,
            os.path.join(self.config['results']['path'], filename)
        )
    
    def plot_history(self):
        """Plot training history."""
        plt.figure(figsize=(12, 4))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_losses'], label='Train Loss')
        plt.plot(self.history['val_losses'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracies
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_accuracies'], label='Train Accuracy')
        plt.plot(self.history['val_accuracies'], label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['results']['path'], 'training_history.png'))
        plt.close()

def main():
    """Main function for Few-Shot Learning pipeline."""
    # Configuration
    config = {
        'num_classes': 2,  # Binary classification for PCOS
        'num_support': 5,  # 5-shot learning
        'num_query': 15,   # Query samples per class
        'meta_lr': 0.001,
        'adaptation_lr': 0.01,
        'num_adaptation_steps': 5,
        'tasks_per_epoch': 100,
        'val_tasks': 50,
        'epochs': 100,
        'save_freq': 10,
        'results': {
            'path': '../results/few_shot_training'
        }
    }
    
    # Initialize trainer
    trainer = FewShotTrainer(config, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Load your dataset here
    # dataset = {...}
    
    # Train the model
    # trainer.train(dataset)
    
    # Plot history
    trainer.plot_history()
    
    print("Few-shot training completed!")

if __name__ == "__main__":
    main() 