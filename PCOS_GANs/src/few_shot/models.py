"""
Few-Shot Learning model architectures using MAML for PCOS detection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
import numpy as np

class ConvBlock(nn.Module):
    """Convolutional block for feature extraction."""
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class MAMLModel(nn.Module):
    """MAML model for Few-Shot Learning."""
    def __init__(self, config: Dict):
        super(MAMLModel, self).__init__()
        self.config = config
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64),
            ConvBlock(64, 64),
            ConvBlock(64, 64)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64 * 16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, config['num_classes'])
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class MAML:
    """Model-Agnostic Meta-Learning implementation."""
    def __init__(self, config: Dict):
        """
        Initialize MAML trainer.
        
        Args:
            config (Dict): Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = MAMLModel(config).to(self.device)
        
        # Initialize optimizer
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['meta_lr']
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def adapt(self, 
             support_images: torch.Tensor,
             support_labels: torch.Tensor,
             num_adaptation_steps: int) -> nn.Module:
        """
        Adapt the model to the support set.
        
        Args:
            support_images (torch.Tensor): Support set images
            support_labels (torch.Tensor): Support set labels
            num_adaptation_steps (int): Number of gradient steps for adaptation
            
        Returns:
            nn.Module: Adapted model
        """
        # Create a clone of the model for adaptation
        adapted_model = MAMLModel(self.config).to(self.device)
        adapted_model.load_state_dict(self.model.state_dict())
        
        # Create optimizer for adaptation
        optimizer = torch.optim.SGD(
            adapted_model.parameters(),
            lr=self.config['adaptation_lr']
        )
        
        # Adapt the model
        for _ in range(num_adaptation_steps):
            predictions = adapted_model(support_images)
            loss = self.criterion(predictions, support_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return adapted_model
    
    def meta_train_step(self,
                       tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> float:
        """
        Perform one meta-training step.
        
        Args:
            tasks (List[Tuple]): List of (support_x, support_y, query_x, query_y) for each task
            
        Returns:
            float: Meta-training loss
        """
        meta_loss = 0.0
        
        for support_x, support_y, query_x, query_y in tasks:
            # Adapt the model to the support set
            adapted_model = self.adapt(
                support_x,
                support_y,
                self.config['num_adaptation_steps']
            )
            
            # Evaluate on query set
            predictions = adapted_model(query_x)
            task_loss = self.criterion(predictions, query_y)
            
            meta_loss += task_loss
        
        # Update meta-parameters
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item() / len(tasks)
    
    def evaluate(self,
                support_x: torch.Tensor,
                support_y: torch.Tensor,
                query_x: torch.Tensor,
                query_y: torch.Tensor) -> Tuple[float, float]:
        """
        Evaluate the model on a task.
        
        Args:
            support_x (torch.Tensor): Support set images
            support_y (torch.Tensor): Support set labels
            query_x (torch.Tensor): Query set images
            query_y (torch.Tensor): Query set labels
            
        Returns:
            Tuple[float, float]: Loss and accuracy
        """
        # Adapt the model to the support set
        adapted_model = self.adapt(
            support_x,
            support_y,
            self.config['num_adaptation_steps']
        )
        
        # Evaluate on query set
        with torch.no_grad():
            predictions = adapted_model(query_x)
            loss = self.criterion(predictions, query_y)
            
            # Calculate accuracy
            _, predicted = torch.max(predictions.data, 1)
            accuracy = (predicted == query_y).float().mean().item()
        
        return loss.item(), accuracy 