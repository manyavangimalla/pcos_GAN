"""
Model training script for CNNs, SVMs, Logistic Regression, Decision Trees, etc.
"""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

from src.utils.helpers import load_data
from src.utils.visualize import plot_metrics, plot_comparison

class ModelTrainer:
    def __init__(self, model: Any, config: Dict, model_type: str = 'pytorch'):
        """
        Initialize the model trainer.

        Args:
            model (Any): The model to train (can be PyTorch or scikit-learn).
            config (Dict): Configuration dictionary.
            model_type (str): Type of model ('pytorch' or 'sklearn').
        """
        self.model = model
        self.config = config
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() and model_type == 'pytorch' else 'cpu')
        if model_type == 'pytorch':
            self.model.to(self.device)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Train the model.
        """
        if self.model_type == 'pytorch':
            self._train_pytorch(train_loader, val_loader)
        else:
            self._train_sklearn(train_loader)

    def _train_pytorch(self, train_loader: DataLoader, val_loader: DataLoader):
        """Train a PyTorch model."""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['lr'])
        
        for epoch in range(self.config['epochs']):
            self.model.train()
            train_loss, train_correct = 0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                train_correct += torch.sum(preds == labels.data)

            epoch_loss = train_loss / len(train_loader)
            epoch_acc = train_correct.double() / len(train_loader.dataset)
            self.history['train_loss'].append(epoch_loss)
            self.history['train_acc'].append(epoch_acc.item())
            
            val_loss, val_acc = self.evaluate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{self.config['epochs']}: "
                  f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    def _train_sklearn(self, train_loader: DataLoader):
        """Train a scikit-learn model."""
        # Scikit-learn models are typically trained on the full dataset at once
        X_train, y_train = next(iter(train_loader))
        self.model.fit(X_train.numpy(), y_train.numpy())
        # Sklearn training doesn't have epochs, so history is not populated here
        print("Scikit-learn model training complete.")

    def evaluate(self, data_loader: DataLoader) -> tuple[float, float]:
        """
        Evaluate the model.
        """
        if self.model_type == 'pytorch':
            return self._evaluate_pytorch(data_loader)
        else:
            return self._evaluate_sklearn(data_loader)

    def _evaluate_pytorch(self, data_loader: DataLoader) -> tuple[float, float]:
        """Evaluate a PyTorch model."""
        self.model.eval()
        val_loss, val_correct = 0, 0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
        
        epoch_loss = val_loss / len(data_loader)
        epoch_acc = val_correct.double() / len(data_loader.dataset)
        return epoch_loss, epoch_acc.item()

    def _evaluate_sklearn(self, data_loader: DataLoader) -> tuple[float, float]:
        """Evaluate a scikit-learn model."""
        X_test, y_test = next(iter(data_loader))
        preds = self.model.predict(X_test.numpy())
        acc = accuracy_score(y_test.numpy(), preds)
        # Loss is not typically calculated for sklearn models in this context
        return 0.0, acc
        
    def plot_history(self, save_path: str):
        """
        Plot training and validation history.
        """
        if not self.history['train_loss']: # Sklearn case
            print("No history to plot for scikit-learn model.")
            return

        plot_metrics(self.history, ['train_loss', 'val_loss'], 'Loss', 'Epoch', os.path.join(save_path, 'loss_plot.png'))
        plot_metrics(self.history, ['train_acc', 'val_acc'], 'Accuracy', 'Epoch', os.path.join(save_path, 'accuracy_plot.png'))


def main():
    """Main function for model training pipeline."""
    # Example config
    config = {
        'lr': 0.001,
        'epochs': 10,
        'batch_size': 32,
        'output_dir': 'results/cnn_training'
    }

    # Create dummy data for demonstration
    X = np.random.rand(100, 3, 32, 32).astype('float32')
    y = np.random.randint(0, 2, 100).astype('int64')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])

    # Example with a simple PyTorch CNN
    # Define a simple CNN for demonstration
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(16 * 16 * 16, 2)
        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = x.view(-1, 16 * 16 * 16)
            x = self.fc1(x)
            return x

    cnn_model = SimpleCNN()
    
    os.makedirs(config['output_dir'], exist_ok=True)

    trainer = ModelTrainer(cnn_model, config, model_type='pytorch')
    trainer.train(train_loader, test_loader)
    trainer.plot_history(config['output_dir'])

    print("Model training completed!")


if __name__ == "__main__":
    main() 