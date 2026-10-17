import torch

def interpolate_latent_space(generator, latent_dim, num_points=100, device='cpu'):
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

# Usage:
# interpolated_data = interpolate_latent_space(generator, latent_dim, device=device)
# inverse_transformed_data = scaler.inverse_transform(interpolated_data)