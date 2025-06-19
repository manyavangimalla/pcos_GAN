"""
Evaluation script for model performance metrics.
"""
import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix
)

class ModelEvaluator:
    def __init__(self, config: Dict):
        """
        Initialize model evaluator.
        
        Args:
            config (Dict): Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
    
    def evaluate_classifier(self,
                          model: torch.nn.Module,
                          test_loader: torch.utils.data.DataLoader) -> Dict:
        """
        Evaluate a classifier model.
        
        Args:
            model (torch.nn.Module): The model to evaluate
            test_loader (DataLoader): DataLoader for test data
            
        Returns:
            Dict: Dictionary containing evaluation metrics
        """
        model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds),
            'recall': recall_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds)
        }
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        metrics['auc'] = auc(fpr, tpr)
        metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr}
        
        # Calculate confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(all_labels, all_preds)
        
        self.results = metrics
        return metrics
    
    def evaluate_gan(self,
                    generator: torch.nn.Module,
                    real_images: torch.Tensor,
                    num_samples: int = 1000) -> Dict:
        """
        Evaluate a GAN model.
        
        Args:
            generator (torch.nn.Module): The generator model
            real_images (torch.Tensor): Sample of real images
            num_samples (int): Number of samples to generate
            
        Returns:
            Dict: Dictionary containing evaluation metrics
        """
        generator.eval()
        
        # Generate samples
        z = torch.randn(num_samples, self.config['latent_dim']).to(self.device)
        with torch.no_grad():
            generated_images = generator(z)
        
        # Calculate FID score (simplified version)
        fid_score = self._calculate_fid(real_images, generated_images)
        
        metrics = {
            'fid_score': fid_score
        }
        
        self.results = metrics
        return metrics
    
    def _calculate_fid(self, real_images: torch.Tensor, generated_images: torch.Tensor) -> float:
        """
        Calculate Fréchet Inception Distance (FID) score.
        This is a simplified version - in practice, you'd want to use a pre-trained
        inception model and calculate the actual FID score.
        
        Args:
            real_images (torch.Tensor): Real images
            generated_images (torch.Tensor): Generated images
            
        Returns:
            float: FID score
        """
        # Convert to numpy
        real = real_images.cpu().numpy()
        generated = generated_images.cpu().numpy()
        
        # Calculate mean and covariance for real and generated images
        mu_real = np.mean(real, axis=0)
        sigma_real = np.cov(real, rowvar=False)
        
        mu_gen = np.mean(generated, axis=0)
        sigma_gen = np.cov(generated, rowvar=False)
        
        # Calculate FID
        diff = mu_real - mu_gen
        covmean = np.sqrt(sigma_real.dot(sigma_gen))
        
        fid = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
        
        return float(fid)
    
    def plot_results(self, output_dir: str):
        """
        Plot evaluation results.
        
        Args:
            output_dir (str): Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if 'confusion_matrix' in self.results:
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                self.results['confusion_matrix'],
                annot=True,
                fmt='d',
                cmap='Blues'
            )
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
            plt.close()
        
        if 'roc_curve' in self.results:
            # Plot ROC curve
            plt.figure(figsize=(8, 6))
            plt.plot(
                self.results['roc_curve']['fpr'],
                self.results['roc_curve']['tpr'],
                label=f"AUC = {self.results['auc']:.3f}"
            )
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
            plt.close()
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame([{
            k: v for k, v in self.results.items()
            if isinstance(v, (int, float)) and k not in ['roc_curve', 'confusion_matrix']
        }])
        metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)

def main():
    """Main function for evaluation pipeline."""
    config = {
        'latent_dim': 100,
        'output_dir': '../results/evaluation'
    }
    
    evaluator = ModelEvaluator(config)
    
    # Example usage:
    # 1. For classifier evaluation
    # metrics = evaluator.evaluate_classifier(model, test_loader)
    
    # 2. For GAN evaluation
    # metrics = evaluator.evaluate_gan(generator, real_images)
    
    # 3. Plot results
    # evaluator.plot_results(config['output_dir'])
    
    print("Evaluation completed!")

if __name__ == "__main__":
    main() 