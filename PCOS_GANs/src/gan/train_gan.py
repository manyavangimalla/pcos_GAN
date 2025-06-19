"""
GAN training script for synthetic ultrasound image generation.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from datetime import datetime

from .models import Generator, Discriminator

class GANTrainer:
    def __init__(self, config: Dict):
        """
        Initialize GAN trainer with configuration.
        
        Args:
            config (Dict): Configuration dictionary containing hyperparameters
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize networks
        self.generator = Generator(
            latent_dim=config['latent_dim'],
            channels=config['channels']
        ).to(self.device)
        
        self.discriminator = Discriminator(
            channels=config['channels']
        ).to(self.device)
        
        # Initialize optimizers
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=config['lr'],
            betas=(config['b1'], config['b2'])
        )
        
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=config['lr'],
            betas=(config['b1'], config['b2'])
        )
        
        # Loss function
        self.adversarial_loss = nn.BCELoss()
        
        # Initialize training history
        self.history = {
            'D_losses': [],
            'G_losses': [],
            'real_scores': [],
            'fake_scores': []
        }
        
    def train_step(self, real_imgs: torch.Tensor) -> Tuple[float, float]:
        """
        Training step for one batch.
        
        Args:
            real_imgs (torch.Tensor): Batch of real images
            
        Returns:
            Tuple[float, float]: Discriminator and Generator losses
        """
        batch_size = real_imgs.size(0)
        
        # Ground truths
        valid = torch.ones(batch_size, 1).to(self.device)
        fake = torch.zeros(batch_size, 1).to(self.device)
        
        # -----------------
        #  Train Generator
        # -----------------
        self.optimizer_G.zero_grad()
        
        # Sample noise as generator input
        z = torch.randn(batch_size, self.config['latent_dim']).to(self.device)
        
        # Generate a batch of images
        gen_imgs = self.generator(z)
        
        # Loss measures generator's ability to fool the discriminator
        g_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid)
        
        g_loss.backward()
        self.optimizer_G.step()
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        self.optimizer_D.zero_grad()
        
        # Measure discriminator's ability to classify real from generated samples
        real_loss = self.adversarial_loss(self.discriminator(real_imgs), valid)
        fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        
        d_loss.backward()
        self.optimizer_D.step()
        
        return d_loss.item(), g_loss.item()
    
    def train(self, dataloader: DataLoader, epochs: int):
        """
        Train the GAN.
        
        Args:
            dataloader (DataLoader): DataLoader for training data
            epochs (int): Number of epochs to train
        """
        # Create output directory for generated images
        os.makedirs(os.path.join('PCOS_GANs', self.config['output_dir']), exist_ok=True)
        
        for epoch in range(epochs):
            for i, (real_imgs,) in enumerate(dataloader):
                real_imgs = real_imgs.to(self.device)
                
                # Train for one step
                d_loss, g_loss = self.train_step(real_imgs)
                
                # Save losses
                self.history['D_losses'].append(d_loss)
                self.history['G_losses'].append(g_loss)
                
                # Print progress
                if i % self.config['print_freq'] == 0:
                    print(
                        f"[Epoch {epoch}/{epochs}] "
                        f"[Batch {i}/{len(dataloader)}] "
                        f"[D loss: {d_loss:.4f}] "
                        f"[G loss: {g_loss:.4f}]"
                    )
                
                # Save generated images periodically
                batches_done = epoch * len(dataloader) + i
                if batches_done % self.config['sample_interval'] == 0:
                    self.save_images(batches_done)
            
            # Save model checkpoints
            if epoch % self.config['save_freq'] == 0:
                self.save_checkpoint(epoch)
    
    def save_images(self, batches_done: int):
        """
        Save a grid of generated images.
        
        Args:
            batches_done (int): Current training iteration
        """
        # Generate images
        z = torch.randn(16, self.config['latent_dim']).to(self.device)
        gen_imgs = self.generator(z)
        
        # Save images
        save_image(
            gen_imgs.data[:16],
            os.path.join(
                os.path.join('PCOS_GANs', self.config['output_dir']),
                f"images_{batches_done}.png"
            ),
            nrow=4,
            normalize=True
        )
    
    def save_checkpoint(self, epoch: int):
        """
        Save model checkpoints.
        
        Args:
            epoch (int): Current epoch
        """
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'history': self.history
        }
        
        torch.save(
            checkpoint,
            os.path.join(
                os.path.join('PCOS_GANs', self.config['output_dir']),
                f"checkpoint_epoch_{epoch}.pth"
            )
        )
    
    def plot_losses(self):
        """Plot training losses."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['D_losses'], label='Discriminator loss')
        plt.plot(self.history['G_losses'], label='Generator loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(os.path.join('PCOS_GANs', self.config['output_dir']), 'loss_plot.png'))
        plt.close()

def main():
    """Main function for GAN training pipeline."""
    # Configuration
    config = {
        'latent_dim': 100,
        'channels': 3,
        'lr': 0.0002,
        'b1': 0.5,
        'b2': 0.999,
        'batch_size': 64,
        'epochs': 200,
        'print_freq': 100,
        'sample_interval': 400,
        'save_freq': 10,
        'output_dir': 'results/gan_training'
    }
    
    # Initialize trainer
    trainer = GANTrainer(config)
    
    # Load your dataset here
    # Create a dummy dataloader for demonstration
    dummy_data = torch.randn(128, config['channels'], 32, 32)
    dummy_dataset = torch.utils.data.TensorDataset(dummy_data)
    dataloader = torch.utils.data.DataLoader(dummy_dataset, batch_size=config['batch_size'])
    
    # Train the model
    trainer.train(dataloader, config['epochs'])
    
    # Plot losses
    trainer.plot_losses()
    
    print("GAN training completed!")

if __name__ == "__main__":
    main() 