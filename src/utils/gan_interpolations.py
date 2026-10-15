"""Latent-space interpolation for a trained generator."""
import torch
import numpy as np


def interpolate_latent_space(generator: torch.nn.Module, latent_dim: int, num_points: int = 100, device: str = 'cpu') -> np.ndarray:
    """Generate a trajectory along a line between two random latent vectors.

    Returns a float array of shape (num_points * seq_len, 3).
    """
    generator.eval()
    with torch.no_grad():
        # Generate two random points in latent space
        z1 = torch.randn(1, latent_dim, device=device)
        z2 = torch.randn(1, latent_dim, device=device)
        
        # Generate points along the line between z1 and z2
        alpha = torch.linspace(0, 1, num_points, device=device).unsqueeze(1)
        interpolated_z = alpha * z1 + (1 - alpha) * z2
        
        # Generate data from the interpolated latent points
        interpolated_data = generator(interpolated_z).cpu().numpy()
    
    # Each interpolated point produces (seq_len, 3); flatten segments into trajectory
    return interpolated_data.reshape(-1, 3)
