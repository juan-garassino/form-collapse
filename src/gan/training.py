import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Any, Tuple
import logging
from .models import create_models
from ..utils.visualization import plot_attractor, plot_phase_space, plot_time_series
from ..utils.svg_gcode import save_svg, generate_gcode

logger = logging.getLogger(__name__)

def preprocess_data(results: Dict[str, np.ndarray], seq_len: int = 64) -> torch.Tensor:
    """Preprocess the simulation results for GAN training.

    Windows trajectories into overlapping segments of shape (num_segments, seq_len, 3)
    with stride seq_len // 2.
    """
    all_data = np.concatenate(list(results.values()), axis=0)
    # Normalize the data
    data_mean = np.mean(all_data, axis=0)
    data_std = np.std(all_data, axis=0)
    normalized_data = (all_data - data_mean) / data_std

    # Window into overlapping segments
    stride = seq_len // 2
    num_points = normalized_data.shape[0]
    segments = []
    for start in range(0, num_points - seq_len + 1, stride):
        segments.append(normalized_data[start:start + seq_len])

    segments = np.array(segments)  # (num_segments, seq_len, 3)
    logger.info(f"Created {len(segments)} trajectory segments of length {seq_len}")
    return torch.FloatTensor(segments)

def train_gan(generator: nn.Module, discriminator: nn.Module, dataloader: DataLoader,
              num_epochs: int, latent_dim: int, device: torch.device) -> Tuple[nn.Module, nn.Module]:
    """Train the GAN and return the trained generator and discriminator."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    logger.info(f"Starting GAN training for {num_epochs} epochs")

    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            real_data = batch[0].to(device)
            batch_size = real_data.size(0)

            # Train Discriminator
            optimizer_D.zero_grad()

            label = torch.full((batch_size,), 1, dtype=torch.float, device=device)
            output = discriminator(real_data).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()

            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_data = generator(noise)
            label.fill_(0)
            output = discriminator(fake_data.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()

            errD = errD_real + errD_fake
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            label.fill_(1)
            output = discriminator(fake_data).view(-1)
            errG = criterion(output, label)
            errG.backward()
            optimizer_G.step()

        logger.info(f'[Epoch {epoch+1}/{num_epochs}] Loss_D: {errD.item():.4f}, Loss_G: {errG.item():.4f}')

    logger.info("GAN training completed")
    return generator, discriminator

def generate_samples(generator: nn.Module, latent_dim: int, num_samples: int, device: torch.device) -> np.ndarray:
    """Generate trajectory segments and concatenate them into a continuous trajectory."""
    with torch.no_grad():
        noise = torch.randn(num_samples, latent_dim, device=device)
        generated_segments = generator(noise).cpu().numpy()  # (num_samples, seq_len, 3)
    # Concatenate segments into a continuous trajectory
    trajectory = generated_segments.reshape(-1, 3)  # (num_samples * seq_len, 3)
    return trajectory

def train_gan_on_results(results: Dict[str, np.ndarray], device: torch.device, config: Dict[str, Any], output_dir: str) -> None:
    """Train GAN on the simulation results and generate new data."""
    logger.info("Starting GAN training on simulation results")

    seq_len = config['gan_params'].get('seq_len', 64)

    # Preprocess data
    tensor_data = preprocess_data(results, seq_len=seq_len)
    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=config['gan_params']['batch_size'], shuffle=True)

    # Create and train GAN
    generator, discriminator = create_models(config['gan_params']['latent_dim'], seq_len=seq_len)
    generator.to(device)
    discriminator.to(device)

    generator, discriminator = train_gan(generator, discriminator, dataloader,
                                         num_epochs=config['gan_params']['num_epochs'],
                                         latent_dim=config['gan_params']['latent_dim'],
                                         device=device)

    # Generate samples
    num_samples = 1000
    generated_data = generate_samples(generator, config['gan_params']['latent_dim'], num_samples, device)
    # generated_data is now (num_samples * seq_len, 3)

    # Denormalize the generated data
    all_data = np.concatenate(list(results.values()), axis=0)
    data_mean = np.mean(all_data, axis=0)
    data_std = np.std(all_data, axis=0)
    denormalized_data = generated_data * data_std + data_mean

    # Visualize and save generated data
    logger.info("Saving GAN-generated data visualizations")
    plot_attractor("gan_generated", denormalized_data, output_dir, smooth=True)
    plot_phase_space("gan_generated", denormalized_data, output_dir, smooth=True)
    plot_time_series("gan_generated", denormalized_data, output_dir, smooth=True)

    # Save SVG and G-code
    save_svg(denormalized_data[:, :2], "gan_generated_attractor", output_dir)
    generate_gcode(denormalized_data, "gan_generated_attractor", output_dir)

    logger.info(f"GAN-generated data saved in the '{output_dir}' folder")
