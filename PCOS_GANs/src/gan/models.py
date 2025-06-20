"""
GAN model architectures for PCOS ultrasound image generation.
"""
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        """
        Generator network for GAN.
        
        Args:
            latent_dim (int): Dimension of the latent space
            output_dim (int): Dimension of the output space
        """
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        """
        Forward pass of the generator.
        
        Args:
            z (torch.Tensor): Latent vector
            
        Returns:
            torch.Tensor: Generated image
        """
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        """
        Discriminator network for GAN.
        
        Args:
            input_dim (int): Dimension of the input space
        """
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        """
        Forward pass of the discriminator.
        
        Args:
            img (torch.Tensor): Input image
            
        Returns:
            torch.Tensor: Probability of image being real
        """
        return self.model(img) 