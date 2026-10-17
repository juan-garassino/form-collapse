import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from typing import Dict
import logging
from scipy.signal import welch, savgol_filter
from scipy.interpolate import splprep, splev

logger = logging.getLogger(__name__)

# Set up Seaborn style
sns.set_theme(style="white")
plt.rcParams['axes.edgecolor'] = '0.2'
plt.rcParams['axes.linewidth'] = 0.5

def save_png(fig: plt.Figure, filename: str, output_dir: str) -> None:
    png_dir = os.path.join(output_dir, 'png')
    os.makedirs(png_dir, exist_ok=True)
    full_path = os.path.join(png_dir, f"{filename}.png")
    try:
        fig.savefig(full_path, bbox_inches='tight', dpi=300)
        logger.info(f"Saved plot as {full_path}")
    except Exception as e:
        logger.error(f"Failed to save plot: {str(e)}")

def min_max_scale(data: np.ndarray) -> np.ndarray:
    """Apply min-max scaling to the entire array."""
    min_val = np.min(data)  # Global minimum
    max_val = np.max(data)  # Global maximum
    return (data - min_val) / (max_val - min_val)

def smooth_data(data: np.ndarray, smooth: bool = False, resolution: int = 10000) -> np.ndarray:
    if not smooth:
        return data
    
    x, y, z = data.T
    
    try:
        # Try spline interpolation first with increased resolution
        tck, u = splprep([x, y, z], s=0, k=3)
        u_new = np.linspace(0, 1, resolution)
        new_points = splev(u_new, tck)
        return np.array(new_points).T
    except Exception as e:
        logger.warning(f"Spline interpolation failed: {str(e)}. Using Savitzky-Golay filter instead.")
        
        # Fallback to Savitzky-Golay filter with adjusted parameters
        window_length = min(len(x) // 20 * 2 + 1, 101)  # Increased window length, must be odd
        poly_order = min(3, window_length - 1)  # Must be less than window_length
        
        smoothed_x = savgol_filter(x, window_length, poly_order)
        smoothed_y = savgol_filter(y, window_length, poly_order)
        smoothed_z = savgol_filter(z, window_length, poly_order)
        
        # Interpolate the Savitzky-Golay filtered data to increase resolution
        t = np.linspace(0, 1, len(x))
        t_new = np.linspace(0, 1, resolution)
        smoothed_x = np.interp(t_new, t, smoothed_x)
        smoothed_y = np.interp(t_new, t, smoothed_y)
        smoothed_z = np.interp(t_new, t, smoothed_z)
        
        return np.column_stack((smoothed_x, smoothed_y, smoothed_z))

def remove_top_right_axes(ax: plt.Axes) -> None:
    """Remove the top and right axes from a given Axes object."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def set_equal_aspect(ax):
    """Set equal scaling for 3D plots."""
    limits = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
    center = np.mean(limits, axis=1)
    max_range = np.ptp(limits, axis=1).max() / 2

    for axis, ctr in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], center):
        axis([ctr - max_range, ctr + max_range])

def set_axis_limits_with_margin(ax, data, margin=0.1):
    """Set axis limits with margins for 2D plots."""
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    
    x_margin = (max_vals[0] - min_vals[0]) * margin
    y_margin = (max_vals[1] - min_vals[1]) * margin
    
    ax.set_xlim(min_vals[0] - x_margin, max_vals[0] + x_margin)
    ax.set_ylim(min_vals[1] - y_margin, max_vals[1] + y_margin)

def plot_attractor(name: str, data: np.ndarray, output_dir: str, smooth: bool = False) -> None:
    """Plot a single attractor in 3D and 2D projections with min-max scaling and margins."""
    logger.info(f"Plotting attractor: {name}")

    try:
        smooth_data_points = smooth_data(data, smooth)
        scaled_data = min_max_scale(smooth_data_points)
    except Exception as e:
        logger.error(f"Data processing failed for {name}: {str(e)}. Using original data.")
        scaled_data = min_max_scale(data)

    try:
        # Compute the min and max values of the scaled data
        data_min = np.min(scaled_data, axis=0)
        data_max = np.max(scaled_data, axis=0)
        margin = 0.1  # Add a margin of 0.1 to each axis
        
        limits_min = data_min - margin
        limits_max = data_max + margin

        # 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(scaled_data[:, 0], scaled_data[:, 1], scaled_data[:, 2], lw=0.5)
        ax.set_title(f'{name} Attractor (3D)', fontsize=16)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)

        # Set limits with margins
        ax.set_xlim(limits_min[0], limits_max[0])
        ax.set_ylim(limits_min[1], limits_max[1])
        ax.set_zlim(limits_min[2], limits_max[2])

        # Ensure equal aspect ratio
        set_equal_aspect(ax)

        ax.grid(False)
        save_png(fig, f"{name}_attractor_3d", output_dir)
        plt.close(fig)

        # 2D projections with margin and equal scaling
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # XY Projection
        ax1.plot(scaled_data[:, 0], scaled_data[:, 1], lw=0.5)
        ax1.set_title(f'{name} (XY Projection)', fontsize=14)
        ax1.set_xlabel('X', fontsize=12)
        ax1.set_ylabel('Y', fontsize=12)
        ax1.set_xlim(limits_min[0], limits_max[0])
        ax1.set_ylim(limits_min[1], limits_max[1])
        remove_top_right_axes(ax1)

        # XZ Projection
        ax2.plot(scaled_data[:, 0], scaled_data[:, 2], lw=0.5)
        ax2.set_title(f'{name} (XZ Projection)', fontsize=14)
        ax2.set_xlabel('X', fontsize=12)
        ax2.set_ylabel('Z', fontsize=12)
        ax2.set_xlim(limits_min[0], limits_max[0])
        ax2.set_ylim(limits_min[2], limits_max[2])
        remove_top_right_axes(ax2)

        # YZ Projection
        ax3.plot(scaled_data[:, 1], scaled_data[:, 2], lw=0.5)
        ax3.set_title(f'{name} (YZ Projection)', fontsize=14)
        ax3.set_xlabel('Y', fontsize=12)
        ax3.set_ylabel('Z', fontsize=12)
        ax3.set_xlim(limits_min[1], limits_max[1])
        ax3.set_ylim(limits_min[2], limits_max[2])
        remove_top_right_axes(ax3)

        plt.tight_layout()
        save_png(fig, f"{name}_attractor_2d_projections", output_dir)
        plt.close(fig)
    except Exception as e:
        logger.error(f"Failed to plot attractor {name}: {str(e)}")

def create_summary_plot(results: Dict[str, np.ndarray], output_dir: str, smooth: bool = False) -> None:
    """Create a summary plot of all attractors with min-max scaling."""
    logger.info("Creating summary plot")
    num_attractors = len(results)
    
    if num_attractors == 0:
        logger.warning("No successful simulations to plot. Skipping summary plot.")
        return
    
    rows = int(np.ceil(np.sqrt(num_attractors)))
    cols = int(np.ceil(num_attractors / rows))

    try:
        fig = plt.figure(figsize=(5*cols, 5*rows))
        
        for i, (name, data) in enumerate(results.items()):
            try:
                smooth_data_points = smooth_data(data, smooth)
                scaled_data = min_max_scale(smooth_data_points)
            except Exception as e:
                logger.error(f"Data processing failed for {name}: {str(e)}. Using original data.")
                scaled_data = min_max_scale(data)
            
            ax = fig.add_subplot(rows, cols, i+1, projection='3d')
            ax.plot(scaled_data[:, 0], scaled_data[:, 1], scaled_data[:, 2], lw=0.5)
            ax.set_title(name, fontsize=12)
            ax.set_xlabel('X', fontsize=10)
            ax.set_ylabel('Y', fontsize=10)
            ax.set_zlabel('Z', fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(0, 1)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.grid(False)

        plt.tight_layout()
        save_png(fig, "summary_plot", output_dir)
        plt.close(fig)
    except Exception as e:
        logger.error(f"Failed to create summary plot: {str(e)}")

def plot_phase_space(name: str, data: np.ndarray, output_dir: str, smooth: bool = False) -> None:
    """Plot the phase space of an attractor with min-max scaling and margins."""
    logger.info(f"Plotting phase space for attractor: {name}")

    try:
        smooth_data_points = smooth_data(data, smooth)
        scaled_data = min_max_scale(smooth_data_points)
    except Exception as e:
        logger.error(f"Data processing failed for {name}: {str(e)}. Using original data.")
        scaled_data = min_max_scale(data)

    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.kdeplot(x=scaled_data[:, 0], y=scaled_data[:, 1], cmap="YlGnBu", fill=True, cbar=True)

        # Calculate margins
        x_margin = 0.1 * (scaled_data[:, 0].max() - scaled_data[:, 0].min())
        y_margin = 0.1 * (scaled_data[:, 1].max() - scaled_data[:, 1].min())

        # Set limits with margins
        ax.set_xlim(scaled_data[:, 0].min() - x_margin, scaled_data[:, 0].max() + x_margin)
        ax.set_ylim(scaled_data[:, 1].min() - y_margin, scaled_data[:, 1].max() + y_margin)

        ax.set_title(f'{name} Attractor Phase Space', fontsize=16)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        remove_top_right_axes(ax)
        
        plt.tight_layout()
        save_png(fig, f"{name}_phase_space", output_dir)
        plt.close(fig)
    except Exception as e:
        logger.error(f"Failed to plot phase space for attractor {name}: {str(e)}")

def plot_time_series(name: str, data: np.ndarray, output_dir: str, smooth: bool = False) -> None:
    """Plot the time series of an attractor with min-max scaling."""
    logger.info(f"Plotting time series for attractor: {name}")

    try:
        smooth_data_points = smooth_data(data, smooth)
        scaled_data = min_max_scale(smooth_data_points)
    except Exception as e:
        logger.error(f"Data processing failed for {name}: {str(e)}. Using original data.")
        scaled_data = min_max_scale(data)

    try:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        time = np.linspace(0, 1, len(scaled_data))
        
        ax1.plot(time, scaled_data[:, 0], lw=1)
        ax1.set_ylabel('X', fontsize=12)
        ax1.set_ylim(0, 1)
        remove_top_right_axes(ax1)
        
        ax2.plot(time, scaled_data[:, 1], lw=1)
        ax2.set_ylabel('Y', fontsize=12)
        ax2.set_ylim(0, 1)
        remove_top_right_axes(ax2)
        
        ax3.plot(time, scaled_data[:, 2], lw=1)
        ax3.set_ylabel('Z', fontsize=12)
        ax3.set_xlabel('Time', fontsize=12)
        ax3.set_ylim(0, 1)
        remove_top_right_axes(ax3)
        
        fig.suptitle(f'{name} Attractor Time Series', fontsize=16)
        
        plt.tight_layout()
        save_png(fig, f"{name}_time_series", output_dir)
        plt.close(fig)
    except Exception as e:
        logger.error(f"Failed to plot time series for attractor {name}: {str(e)}")

# def animate_3d(name: str, data: np.ndarray, output_dir: str) -> None:
#     """Create a high-quality 3D animation of the attractor with only the line visible."""
#     logger.info(f"Creating 3D animation for attractor: {name}")

#     scaled_data = min_max_scale(data)

#     # Calculate margins
#     x_min, x_max = scaled_data[:, 0].min(), scaled_data[:, 0].max()
#     y_min, y_max = scaled_data[:, 1].min(), scaled_data[:, 1].max()
#     z_min, z_max = scaled_data[:, 2].min(), scaled_data[:, 2].max()

#     margin = 0.1
#     x_range = x_max - x_min
#     y_range = y_max - y_min
#     z_range = z_max - z_min

#     x_margin = x_range * margin
#     y_margin = y_range * margin
#     z_margin = z_range * margin

#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     # Set the limits with margin
#     ax.set_xlim(x_min - x_margin, x_max + x_margin)
#     ax.set_ylim(y_min - y_margin, y_max + y_margin)
#     ax.set_zlim(z_min - z_margin, z_max + z_margin)

#     # Plot initialization
#     line, = ax.plot([], [], [], lw=0.5, color='white')  # White line

#     # Set background color to dark gray and hide axis elements
#     dark_gray = '#333333'
#     ax.set_facecolor(dark_gray)
#     fig.patch.set_facecolor(dark_gray)
#     ax.set_xticks([])  # Remove x-axis ticks
#     ax.set_yticks([])  # Remove y-axis ticks
#     ax.set_zticks([])  # Remove z-axis ticks
#     ax.xaxis.pane.set_visible(False)  # Hide the x-axis pane
#     ax.yaxis.pane.set_visible(False)  # Hide the y-axis pane
#     ax.zaxis.pane.set_visible(False)  # Hide the z-axis pane
#     ax.xaxis.line.set_color((0.0, 0.0, 0.0, 0.0))  # Hide x-axis line
#     ax.yaxis.line.set_color((0.0, 0.0, 0.0, 0.0))  # Hide y-axis line
#     ax.zaxis.line.set_color((0.0, 0.0, 0.0, 0.0))  # Hide z-axis line

#     # Animation init function
#     def init():
#         line.set_data([], [])
#         line.set_3d_properties([])
#         return line,

#     # Animation update function
#     def animate(i):
#         line.set_data(scaled_data[:i, 0], scaled_data[:i, 1])
#         line.set_3d_properties(scaled_data[:i, 2])
#         return line,

#     # Define the number of frames and interval
#     num_frames = min(len(scaled_data), 500)  # Limit to 500 frames
#     frame_interval = len(scaled_data) // num_frames

#     # Create the animation
#     anim = FuncAnimation(fig, animate, init_func=init,
#                          frames=range(0, len(scaled_data), frame_interval),
#                          interval=20, blit=True)

#     # Create output directory if it doesn't exist
#     animation_dir = os.path.join(output_dir, 'animations')
#     os.makedirs(animation_dir, exist_ok=True)

#     # Save the animation in high quality
#     anim.save(os.path.join(animation_dir, f'{name}_3d_animation.mp4'),
#               writer='ffmpeg', fps=30, dpi=300, bitrate=5000)  # High DPI and bitrate for quality

#     plt.close(fig)
#     logger.info(f"3D animation for {name} saved successfully")

def plot_poincare_section(name: str, data: np.ndarray, output_dir: str, plane: str = 'xy', threshold: float = 0.5) -> None:
    """Plot Poincaré section of the attractor with min-max scaling."""
    logger.info(f"Plotting Poincaré section for attractor: {name}")

    scaled_data = min_max_scale(data)

    if plane == 'xy':
        x, y, z = scaled_data[:, 0], scaled_data[:, 1], scaled_data[:, 2]
    elif plane == 'yz':
        x, y, z = scaled_data[:, 1], scaled_data[:, 2], scaled_data[:, 0]
    elif plane == 'xz':
        x, y, z = scaled_data[:, 0], scaled_data[:, 2], scaled_data[:, 1]

    crossing_indices = np.where(np.diff(np.sign(z - threshold)))[0]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(x[crossing_indices], y[crossing_indices], s=1, alpha=0.5)
    ax.set_title(f'{name} Attractor Poincaré Section ({plane.upper()} plane)', fontsize=16)
    ax.set_xlabel(plane[0].upper(), fontsize=12)
    ax.set_ylabel(plane[1].upper(), fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    remove_top_right_axes(ax)

    plt.tight_layout()
    save_png(fig, f"{name}_poincare_section_{plane}", output_dir)
    plt.close(fig)

def plot_bifurcation(name: str, system_func: callable, param_range: np.ndarray, param_name: str, output_dir: str, base_params: dict = None) -> None:
    """Plot bifurcation diagram for a given parameter."""
    from scipy.integrate import solve_ivp
    from scipy.signal import argrelmax

    logger.info(f"Plotting bifurcation diagram for attractor: {name}")

    results = []
    for param in param_range:
        full_params = dict(base_params) if base_params else {}
        full_params[param_name] = param
        try:
            sol = solve_ivp(
                lambda t, y: system_func(y, t, full_params),
                (0, 100), np.random.randn(3) * 0.1,
                method='RK45', t_eval=np.linspace(0, 100, 2000),
                rtol=1e-6, atol=1e-9,
            )
            if sol.success:
                trajectory = sol.y.T
                # Discard transient (first half)
                steady = trajectory[len(trajectory) // 2:, 0]
                # Find local maxima for bifurcation
                maxima_idx = argrelmax(steady, order=5)[0]
                if len(maxima_idx) > 0:
                    results.extend([(param, steady[i]) for i in maxima_idx[-50:]])
        except Exception:
            continue

    if not results:
        logger.warning(f"No bifurcation data collected for {name}")
        return

    results = np.array(results)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(results[:, 0], results[:, 1], ',k', alpha=0.1, markersize=0.1)
    ax.set_title(f'{name} Attractor Bifurcation Diagram', fontsize=16)
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('x (local maxima)', fontsize=12)
    remove_top_right_axes(ax)

    plt.tight_layout()
    save_png(fig, f"{name}_bifurcation_diagram", output_dir)
    plt.close(fig)

def plot_lyapunov_exponent(name: str, lyap_exp: np.ndarray, output_dir: str) -> None:
    """Plot Lyapunov exponent spectrum."""
    logger.info(f"Plotting Lyapunov exponent spectrum for attractor: {name}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(lyap_exp) + 1), lyap_exp, 'o-')
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_title(f'{name} Attractor Lyapunov Exponent Spectrum', fontsize=16)
    ax.set_xlabel('Exponent Index', fontsize=12)
    ax.set_ylabel('Lyapunov Exponent', fontsize=12)
    remove_top_right_axes(ax)

    plt.tight_layout()
    save_png(fig, f"{name}_lyapunov_spectrum", output_dir)
    plt.close(fig)

def plot_power_spectrum(name: str, data: np.ndarray, output_dir: str) -> None:
    """Plot power spectrum of the attractor with min-max scaling."""
    logger.info(f"Plotting power spectrum for attractor: {name}")

    # Normalize the data
    scaled_data = min_max_scale(data)

    # Set nperseg to be the minimum of 256 or the length of the data to avoid mismatch issues
    nperseg = min(256, len(scaled_data))

    # Compute power spectrum for each dimension
    f, Pxx_den = welch(scaled_data, fs=1, nperseg=nperseg, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(f, Pxx_den[:, 0], label='X')
    ax.semilogy(f, Pxx_den[:, 1], label='Y')
    ax.semilogy(f, Pxx_den[:, 2], label='Z')
    ax.set_title(f'{name} Attractor Power Spectrum', fontsize=16)
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_ylabel('Power Spectral Density', fontsize=12)
    ax.legend()
    remove_top_right_axes(ax)

    plt.tight_layout()
    save_png(fig, f"{name}_power_spectrum", output_dir)
    plt.close(fig)


def plot_parameter_heatmap(sweep_results: list, system_name: str, output_dir: str) -> None:
    """Plot parameter sweep results as a scatter colored by classification."""
    logger.info(f"Creating parameter heatmap for {system_name}")

    # Collect param values and classifications
    valid = [r for r in sweep_results if r['data'] is not None]
    if len(valid) < 2:
        logger.warning(f"Not enough valid results for heatmap ({len(valid)})")
        return

    # Find the two most-varied parameters
    skip_keys = {'sim_time', 'sim_steps', 'scale'}
    param_names = [k for k in valid[0]['params'] if k not in skip_keys
                   and isinstance(valid[0]['params'][k], (int, float))]

    if len(param_names) < 2:
        logger.warning("Need at least 2 parameters for heatmap")
        return

    # Compute variance for each param to find most-varied pair
    param_arrays = {k: np.array([r['params'].get(k, 0) for r in valid]) for k in param_names}
    param_vars = {k: np.var(v) for k, v in param_arrays.items() if np.var(v) > 0}
    sorted_params = sorted(param_vars, key=param_vars.get, reverse=True)

    if len(sorted_params) < 2:
        logger.warning("Not enough varying parameters for heatmap")
        return

    px, py = sorted_params[0], sorted_params[1]
    x_vals = param_arrays[px]
    y_vals = param_arrays[py]

    color_map = {
        'strange_attractor': 'green',
        'limit_cycle': 'blue',
        'fixed_point': 'gray',
        'divergent': 'red',
        'failed': 'black',
    }
    colors = [color_map.get(r['classification'], 'black') for r in valid]

    fig, ax = plt.subplots(figsize=(10, 8))
    for cls, color in color_map.items():
        mask = [c == color for c in colors]
        if any(mask):
            ax.scatter(x_vals[mask], y_vals[mask], c=color, label=cls, s=30, alpha=0.7)

    ax.set_xlabel(px, fontsize=12)
    ax.set_ylabel(py, fontsize=12)
    ax.set_title(f'{system_name} Parameter Sweep', fontsize=16)
    ax.legend()
    remove_top_right_axes(ax)

    plt.tight_layout()
    save_png(fig, f"{system_name}_parameter_heatmap", output_dir)
    plt.close(fig)
