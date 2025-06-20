"""
GAN training script for synthetic ultrasound image generation.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .models import Generator, Discriminator

class GANTrainer:
    """
    Trainer for the GAN model.
    """
    def __init__(self, config: Dict, device):
        """
        Initialize GAN trainer with configuration.
        
        Args:
            config (Dict): Configuration dictionary containing hyperparameters
            device: The device to which the models should be moved
        """
        self.config = config
        self.gan_config = self.config['gan']
        self.device = device
        
        # Initialize networks
        self.generator = Generator(
            latent_dim=self.gan_config['latent_dim'],
            output_dim=self.gan_config['input_dim']
        ).to(self.device)
        self.discriminator = Discriminator(
            input_dim=self.gan_config['input_dim']
        ).to(self.device)
        
        # Initialize optimizers
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=self.gan_config['lr'],
            betas=(self.gan_config['b1'], self.gan_config['b2'])
        )
        
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=self.gan_config['lr'],
            betas=(self.gan_config['b1'], self.gan_config['b2'])
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
        z = torch.randn(batch_size, self.gan_config['latent_dim']).to(self.device)
        
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
        Main training loop for the GAN.
        """
        # Use the correct, dynamic experiment path for logs
        log_dir = os.path.join(self.config['results']['path'], 'logs')
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        
        # Fixed noise for consistent visualization
        fixed_noise = torch.randn(
            self.gan_config['batch_size'], self.gan_config['latent_dim'], device=self.device
        )

        for epoch in range(epochs):
            for i, (real_imgs, _) in enumerate(dataloader):
                real_imgs = real_imgs.to(self.device)
                
                # Train for one step
                d_loss, g_loss = self.train_step(real_imgs)
                
                # Save losses
                self.history['D_losses'].append(d_loss)
                self.history['G_losses'].append(g_loss)
                
                # Print progress
                if i % self.gan_config['print_freq'] == 0:
                    print(
                        f"[Epoch {epoch}/{epochs}] "
                        f"[Batch {i}/{len(dataloader)}] "
                        f"[D loss: {d_loss:.4f}] "
                        f"[G loss: {g_loss:.4f}]"
                    )
                
                # Save generated images periodically
                batches_done = epoch * len(dataloader) + i
                if batches_done % self.gan_config['sample_interval'] == 0:
                    self.save_images(batches_done)
                    # Log generated data distribution to TensorBoard
                    with torch.no_grad():
                        gen_data = self.generator(fixed_noise).detach().cpu()
                        writer.add_histogram('gan/generated_data_distribution', gen_data, batches_done)
                
                # Log metrics to TensorBoard
                writer.add_scalar('Loss/Generator', g_loss.item(), batches_done)
                writer.add_scalar('Loss/Discriminator', d_loss, batches_done)
            
            # Save model checkpoints
            if epoch % self.gan_config['save_freq'] == 0:
                self.save_checkpoint(epoch)
            
            # Log model weights at the end of each epoch
            for name, param in self.generator.named_parameters():
                writer.add_histogram(f'Generator/{name}', param.clone().cpu().data.numpy(), epoch)
            for name, param in self.discriminator.named_parameters():
                writer.add_histogram(f'Discriminator/{name}', param.clone().cpu().data.numpy(), epoch)
        
        # Close the writer
        writer.close()
    
    def save_images(self, batches_done: int):
        """
        Save a grid of generated images.
        
        Args:
            batches_done (int): Current training iteration
        """
        # Generate images
        z = torch.randn(16, self.gan_config['latent_dim']).to(self.device)
        gen_imgs = self.generator(z)
        
        # Save images
        save_image(
            gen_imgs.data[:16],
            os.path.join(
                self.config['results']['path'],
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
                self.config['results']['path'],
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
        plt.savefig(os.path.join(self.config['results']['path'], 'loss_plot.png'))
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
    trainer = GANTrainer(config, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
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