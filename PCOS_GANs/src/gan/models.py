"""
GAN model architectures for PCOS ultrasound image generation.
"""
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim: int, channels: int = 3):
        """
        Generator network for GAN.
        
        Args:
            latent_dim (int): Dimension of the latent space
            channels (int): Number of output channels (default: 3 for RGB)
        """
        super(Generator, self).__init__()
        
        # Initial size for upsampling
        self.init_size = 32
        self.latent_dim = latent_dim
        
        # Linear layer to convert latent vector
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.init_size ** 2)
        )
        
        # Convolutional layers for upsampling
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator.
        
        Args:
            z (torch.Tensor): Latent vector
            
        Returns:
            torch.Tensor: Generated image
        """
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, channels: int = 3):
        """
        Discriminator network for GAN.
        
        Args:
            channels (int): Number of input channels (default: 3 for RGB)
        """
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters: int, out_filters: int, bn: bool = True) -> nn.Sequential:
            """Helper function to create a discriminator block."""
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return nn.Sequential(*block)

        self.model = nn.Sequential(
            discriminator_block(channels, 16, bn=False),
            discriminator_block(16, 32),
            discriminator_block(32, 64),
            discriminator_block(64, 128),
        )

        # Calculate size of flattened features
        ds_size = 256 // 2**4
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator.
        
        Args:
            img (torch.Tensor): Input image
            
        Returns:
            torch.Tensor: Probability of image being real
        """
        features = self.model(img)
        features = features.view(features.shape[0], -1)
        validity = self.adv_layer(features)
        return validity 