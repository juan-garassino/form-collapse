"""Conv1d GAN for 3D trajectory segments of shape (batch, seq_len, 3)."""
import torch
import torch.nn as nn
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class Generator(nn.Module):
    """Map latent vectors of shape (batch, latent_dim) to segments (batch, seq_len, 3), values in [-1, 1]."""

    def __init__(self, latent_dim: int, seq_len: int = 64):
        super(Generator, self).__init__()
        self.seq_len = seq_len

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 16),
            nn.BatchNorm1d(256 * 16),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),   # 32 -> 64
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(64, 3, kernel_size=3, stride=1, padding=1),     # 64 -> 64
            nn.Tanh(),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(seq_len)

        logger.info(f"Initialized Generator with latent_dim: {latent_dim}, seq_len: {seq_len}")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return generated segments, float tensor of shape (batch, seq_len, 3)."""
        x = self.fc(z)                    # (batch, 256*16)
        x = x.view(-1, 256, 16)           # (batch, 256, 16)
        x = self.conv(x)                  # (batch, 3, ~seq_len)
        x = self.adaptive_pool(x)         # (batch, 3, seq_len)
        x = x.permute(0, 2, 1)            # (batch, seq_len, 3)
        return x

class Discriminator(nn.Module):
    """Score segments of shape (batch, seq_len, 3) with one raw logit each, shape (batch, 1)."""

    def __init__(self, seq_len: int = 64):
        super(Discriminator, self).__init__()
        self.seq_len = seq_len

        self.conv = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=4, stride=2, padding=1),   # seq_len -> seq_len//2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1), # seq_len//2 -> seq_len//4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Flatten(),
        )

        self.fc = nn.Linear(128 * (seq_len // 4), 1)

        logger.info(f"Initialized Discriminator with seq_len: {seq_len}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (batch, 1) for input of shape (batch, seq_len, 3)."""
        x = x.permute(0, 2, 1)  # (batch, seq_len, 3) -> (batch, 3, seq_len)
        x = self.conv(x)        # (batch, 128 * seq_len//4)
        x = self.fc(x)          # (batch, 1)
        return x

def create_models(latent_dim: int, seq_len: int = 64) -> Tuple[nn.Module, nn.Module]:
    """Create and return instances of Generator and Discriminator."""
    generator = Generator(latent_dim, seq_len=seq_len)
    discriminator = Discriminator(seq_len=seq_len)
    logger.info(f"Created Generator and Discriminator models with latent_dim: {latent_dim}, seq_len: {seq_len}")
    return generator, discriminator
