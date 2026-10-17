import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import os
import logging

logger = logging.getLogger(__name__)

def min_max_scale(data: np.ndarray) -> np.ndarray:
    """Min-max scale the data to the range [0, 1]."""
    return (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

def compute_curvature(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Compute the curvature of a 3D curve."""
    dx = np.gradient(x)
    dy = np.gradient(y)
    dz = np.gradient(z)
    
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    d2z = np.gradient(dz)

    # kappa = |r' x r''| / |r'|^3 (all three cross-product components).
    cross = np.stack([dy * d2z - dz * d2y, dz * d2x - dx * d2z, dx * d2y - dy * d2x])
    numerator = np.linalg.norm(cross, axis=0)
    denominator = (dx**2 + dy**2 + dz**2)**1.5
    # Guard zero-speed samples without biasing small (finely sampled) steps.
    curvature = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float),
                          where=denominator > 0)

    return curvature

def animate_3d(name: str, data: np.ndarray, output_dir: str) -> None:
    """Create a high-quality 3D animation of the attractor with varying line thickness based on curvature."""
    logger.info(f"Creating 3D animation for attractor: {name}")

    scaled_data = min_max_scale(data)

    x = scaled_data[:, 0]
    y = scaled_data[:, 1]
    z = scaled_data[:, 2]

    # Calculate curvature
    curvature = compute_curvature(x, y, z)
    
    # Normalize curvature to a range suitable for line width
    span = curvature.max() - curvature.min()
    # Normalize to [0, 1]; constant curvature would otherwise divide by zero (NaN widths).
    curvature = (curvature - curvature.min()) / span if span > 0 else np.zeros_like(curvature)
    min_line_width = 0.1
    max_line_width = 1.2
    thickness = min_line_width + (max_line_width - min_line_width) * curvature

    # Create the figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Calculate margins
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()

    margin = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    x_margin = x_range * margin
    y_margin = y_range * margin
    z_margin = z_range * margin

    # Set the limits with margin
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_zlim(z_min - z_margin, z_max + z_margin)

    # Set background color to dark gray and hide axis elements
    dark_gray = '#333333'
    ax.set_facecolor(dark_gray)
    fig.patch.set_facecolor(dark_gray)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    ax.xaxis.line.set_color((0.0, 0.0, 0.0, 0.0))
    ax.yaxis.line.set_color((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.line.set_color((0.0, 0.0, 0.0, 0.0))

    fig.patch.set_alpha(0)

    # Animation init function
    def init():
        return []

    # Animation update function
    def animate(i):
        ax.clear()  # Clear previous frames

        # Reapply formatting
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.set_zlim(z_min - z_margin, z_max + z_margin)
        ax.set_facecolor(dark_gray)
        fig.patch.set_facecolor(dark_gray)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.pane.set_visible(False)
        ax.yaxis.pane.set_visible(False)
        ax.zaxis.pane.set_visible(False)
        ax.xaxis.line.set_color((0.0, 0.0, 0.0, 0.0))
        ax.yaxis.line.set_color((0.0, 0.0, 0.0, 0.0))
        ax.zaxis.line.set_color((0.0, 0.0, 0.0, 0.0))

        # Create the line with varying thickness
        segments = [[(x[j-1], y[j-1], z[j-1]), (x[j], y[j], z[j])] for j in range(1, i)]
        line_collection = Line3DCollection(segments, colors='white', linewidths=thickness[:i])
        ax.add_collection(line_collection)
        
        return []

    # Define the number of frames and interval
    num_frames = min(len(scaled_data), 500)
    frame_interval = len(scaled_data) // num_frames

    # Create the animation
    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=range(0, len(scaled_data), frame_interval),
                         interval=20, blit=True)

    # Create output directory if it doesn't exist
    animation_dir = os.path.join(output_dir, 'animations')
    os.makedirs(animation_dir, exist_ok=True)

    # Save the animation in high quality
    anim.save(os.path.join(animation_dir, f'{name}_3d_animation.mp4'),
              writer='ffmpeg', fps=30, dpi=300, bitrate=5000)

    plt.close(fig)
    logger.info(f"3D animation for {name} saved successfully")
