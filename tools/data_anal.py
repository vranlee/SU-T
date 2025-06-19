import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.signal import savgol_filter
from pathlib import Path
import cv2
from PIL import Image
import glob


# Configure default plot settings
def setup_plot_style(font_name='Arial', text_size=12):
    """Configure global plot style settings"""
    plt.rcParams.update({
        'font.family': font_name,
        'font.size': text_size,
        'axes.labelsize': text_size,
        'axes.titlesize': text_size + 2,
        'xtick.labelsize': text_size,
        'ytick.labelsize': text_size,
        'legend.fontsize': text_size,
        'figure.titlesize': text_size + 4
    })

def load_tracking_data(file_path):
    """Load tracking data from file into pandas DataFrame"""
    columns = ['frame', 'id', 'x', 'y', 'width', 'height', 'confidence', 'class', 'visibility']
    # Adjust column names based on your actual data format
    df = pd.read_csv(file_path, header=None, names=columns)
    return df


def analyze_clip(df, clip_name, output_dirs, colors=None, font_name='Arial', text_size=12):
    """Analyze tracking data for a single clip"""
    # Calculate derived metrics
    df = calculate_motion_metrics(df)

    # Various analysis functions
    speed_analysis = analyze_speed(df, clip_name, output_dirs['speed'], colors, font_name, text_size)
    angular_analysis = analyze_angular_velocity(df, clip_name, output_dirs['angular_velocity'],
                                                output_dirs['direction'], colors, font_name, text_size)
    size_analysis = analyze_target_size(df, clip_name, output_dirs['size'], colors, font_name, text_size)
    count_analysis = analyze_target_counts(df, clip_name, output_dirs['counts'], colors, font_name, text_size)
    feature_analysis = analyze_target_features(df, clip_name, output_dirs['features'], colors, font_name, text_size)

    return {
        'speed': speed_analysis,
        'angular': angular_analysis,
        'size': size_analysis,
        'counts': count_analysis,
        'features': feature_analysis
    }

def calculate_motion_metrics(df):
    """Calculate speed, acceleration, angular velocity for each object"""
    # Group by object ID
    result = df.copy()

    # Sort by frame for each object
    result = result.sort_values(['id', 'frame'])

    # Calculate displacement between frames
    result['dx'] = result.groupby('id')['x'].diff()
    result['dy'] = result.groupby('id')['y'].diff()
    result['dt'] = result.groupby('id')['frame'].diff()  # Assuming constant frame rate

    # Calculate speed (pixels per frame)
    result['speed'] = np.sqrt(result['dx'] ** 2 + result['dy'] ** 2) / result['dt']

    # Calculate direction (angle in radians)
    result['angle'] = np.arctan2(result['dy'], result['dx'])

    # Calculate angular velocity (change in angle per frame)
    result['angle_change'] = result.groupby('id')['angle'].diff()

    # Fix angle wrapping (-pi to pi)
    result['angle_change'] = ((result['angle_change'] + np.pi) % (2 * np.pi)) - np.pi

    # Calculate angular velocity
    result['angular_velocity'] = result['angle_change'] / result['dt']

    # Calculate area
    result['area'] = result['width'] * result['height']

    return result

def analyze_speed(df, clip_name, output_dir, colors=None, font_name='Arial', text_size=12):
    """Analyze target speed trends with customizable styling"""
    # Use default colors if none provided
    if colors is None:
        colors = {'hist': 'skyblue', 'line': 'royalblue', 'trend': 'crimson'}

    # Speed distribution
    plt.figure(figsize=(10, 10))
    sns.histplot(df['speed'].dropna(), kde=True, color=colors.get('hist', 'skyblue'))
    plt.xlabel('Speed (pixels/frame)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'speed_distribution_{clip_name}.png'))
    plt.close()

    # Speed trend over time
    plt.figure(figsize=(10, 10))
    speed_by_frame = df.groupby('frame')['speed'].mean().reset_index()
    plt.plot(speed_by_frame['frame'], speed_by_frame['speed'], color=colors.get('line', 'royalblue'))
    # Add smoothed trend line
    if len(speed_by_frame) > 10:
        smoothed = savgol_filter(speed_by_frame['speed'], min(11, len(speed_by_frame) // 2 * 2 + 1), 3)
        plt.plot(speed_by_frame['frame'], smoothed, color=colors.get('trend', 'crimson'), linewidth=2)
    plt.xlabel('Frame')
    plt.ylabel('Avg Speed')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'speed_trend_{clip_name}.png'))
    plt.close()

    # Return summary statistics
    return {
        'mean_speed': df['speed'].mean(),
        'max_speed': df['speed'].max(),
        'min_speed': df['speed'].min(),
        'std_speed': df['speed'].std()
    }


def analyze_angular_velocity(df, clip_name, angular_output_dir, direction_output_dir,
                             colors=None, font_name='Arial', text_size=12):
    """Analyze target angular velocity trends with customizable styling"""
    if colors is None:
        colors = {'hist': 'lightgreen', 'line': 'green', 'trend': 'darkred', 'polar': 'purple'}

    # Angular velocity distribution
    plt.figure(figsize=(10, 10))
    sns.histplot(df['angular_velocity'].dropna(), kde=True, color=colors.get('hist', 'lightgreen'))
    plt.xlabel('Angular Velocity (radians/frame)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(angular_output_dir, f'angular_velocity_distribution_{clip_name}.png'))
    plt.close()

    # Angular velocity trend over time
    plt.figure(figsize=(10, 10))
    angular_by_frame = df.groupby('frame')['angular_velocity'].mean().reset_index()
    plt.plot(angular_by_frame['frame'], angular_by_frame['angular_velocity'], color=colors.get('line', 'green'))
    # Add smoothed trend line
    if len(angular_by_frame) > 10:
        smoothed = savgol_filter(angular_by_frame['angular_velocity'], min(11, len(angular_by_frame) // 2 * 2 + 1), 3)
        plt.plot(angular_by_frame['frame'], smoothed, color=colors.get('trend', 'darkred'), linewidth=2)
    plt.xlabel('Frame')
    plt.ylabel('Avg Angular Velocity')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(angular_output_dir, f'angular_velocity_trend_{clip_name}.png'))
    plt.close()

    # Movement direction distribution (polar plot)
    # Fix: Use plt.subplots() instead of plt.figure() for polar projection
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
    bins = np.linspace(-np.pi, np.pi, 36)  # 10-degree bins
    angles = df['angle'].dropna().values
    ax.hist(angles, bins=bins, color=colors.get('polar', 'purple'))

    # Hide radius axis tick marks
    ax.tick_params(axis='y', colors='none')  # Transparent tick marks
    # Or hide labels
    ax.set_yticklabels([])  # Clear labels

    plt.tight_layout()
    plt.savefig(os.path.join(direction_output_dir, f'direction_distribution_{clip_name}.png'))
    plt.close(fig)

    # Calculate absolute angular velocity for some statistics
    abs_ang_vel = df['angular_velocity'].abs().dropna()

    # Return summary statistics
    return {
        'mean_angular_velocity': df['angular_velocity'].mean(),
        'mean_abs_angular_velocity': abs_ang_vel.mean(),
        'max_angular_velocity': abs_ang_vel.max(),
        'std_angular_velocity': df['angular_velocity'].std(),
        'direction_entropy': calculate_direction_entropy(angles)
    }

def calculate_direction_entropy(angles, num_bins=36):
    """Calculate entropy of movement directions (higher = more uniform directions)"""
    if len(angles) == 0:
        return 0

    hist, _ = np.histogram(angles, bins=np.linspace(-np.pi, np.pi, num_bins + 1))
    hist = hist / hist.sum()
    # Remove zeros to avoid log(0)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log(hist))


def analyze_target_size(df, clip_name, output_dir, colors=None, font_name='Arial', text_size=12):
    """Analyze target size and area statistics with customizable styling"""
    if colors is None:
        colors = {'hist': 'salmon', 'boxplot': 'lightcoral', 'line': 'firebrick'}

    # Area distribution
    plt.figure(figsize=(10, 10))
    sns.histplot(df['area'].dropna(), kde=True, color=colors.get('hist', 'salmon'))
    plt.xlabel('Area (pixels²)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'area_distribution_{clip_name}.png'))
    plt.close()

    # Area by object class (if available)
    if 'class' in df.columns and df['class'].nunique() > 1:
        plt.figure(figsize=(10, 10))
        sns.boxplot(x='class', y='area', data=df, palette=[colors.get('boxplot', 'lightcoral')])
        plt.xlabel('Object Class')
        plt.ylabel('Area (pixels²)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'area_by_class_{clip_name}.png'))
        plt.close()

    # Area change over time
    plt.figure(figsize=(10, 10))
    area_by_frame = df.groupby('frame')['area'].mean().reset_index()
    plt.plot(area_by_frame['frame'], area_by_frame['area'], color=colors.get('line', 'firebrick'))
    plt.xlabel('Frame')
    plt.ylabel('Average Area (pixels²)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'area_trend_{clip_name}.png'))
    plt.close()

    return {
        'mean_area': df['area'].mean(),
        'mean_width': df['width'].mean(),
        'mean_height': df['height'].mean()
    }


def analyze_target_counts(df, clip_name, output_dir, colors=None, font_name='Arial', text_size=12):
    """Analyze number of targets per frame with customizable styling"""
    if colors is None:
        colors = {'line': 'teal'}

    counts_per_frame = df.groupby('frame')['id'].nunique()

    plt.figure(figsize=(10, 10))
    counts_per_frame.plot(color=colors.get('line', 'teal'))
    plt.xlabel('Frame')
    plt.ylabel('Target Count')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'target_counts_{clip_name}.png'))
    plt.close()

    return {
        'max_targets': counts_per_frame.max(),
        'min_targets': counts_per_frame.min(),
        'mean_targets': counts_per_frame.mean(),
        'total_unique_targets': df['id'].nunique()
    }


def analyze_target_features(df, clip_name, output_dir, colors=None, font_name='Arial', text_size=12):
    """Analyze differences between targets with customizable styling"""
    if df['id'].nunique() < 2:
        return {"message": "Not enough unique targets for feature comparison"}

    # Analysis based on available features
    feature_summary = {}

    # Compare size differences between objects
    obj_sizes = df.groupby('id')['area'].mean().reset_index()
    if len(obj_sizes) > 0:
        feature_summary['size_variation'] = obj_sizes['area'].std() / obj_sizes['area'].mean() if obj_sizes[
                                                                                                      'area'].mean() > 0 else 0

    # Compare speed differences between objects
    obj_speeds = df.groupby('id')['speed'].mean().reset_index()
    if len(obj_speeds) > 0 and 'speed' in obj_speeds and obj_speeds['speed'].mean() > 0:
        feature_summary['speed_variation'] = obj_speeds['speed'].std() / obj_speeds['speed'].mean()

    # Feature correlation analysis (for objects with sufficient data)
    features = ['area', 'speed', 'angular_velocity']
    features = [f for f in features if f in df.columns]
    if len(features) >= 2:
        plt.figure(figsize=(8, 6))
        corr_matrix = df[features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'feature_correlation_{clip_name}.png'))
        plt.close()

    return feature_summary

def create_output_directories(output_base_dir='analysis_results', dataset_names=None):
    """Create directory structure for organizing output files by dataset and analysis type"""
    # Create base directory
    os.makedirs(output_base_dir, exist_ok=True)

    # Analysis types to organize by
    analysis_types = ['speed', 'angular_velocity', 'direction', 'size', 'counts', 'features']

    directories = {}

    # If dataset names are provided, create separate folders for each dataset
    if dataset_names:
        for dataset in dataset_names:
            dataset_dir = os.path.join(output_base_dir, dataset)
            os.makedirs(dataset_dir, exist_ok=True)

            dataset_dirs = {}
            for analysis_type in analysis_types:
                path = os.path.join(dataset_dir, analysis_type)
                os.makedirs(path, exist_ok=True)
                dataset_dirs[analysis_type] = path

            directories[dataset] = dataset_dirs
    else:
        # Create default structure without dataset separation
        default_dirs = {}
        for analysis_type in analysis_types:
            path = os.path.join(output_base_dir, analysis_type)
            os.makedirs(path, exist_ok=True)
            default_dirs[analysis_type] = path

        directories["default"] = default_dirs

    return directories


def analyze_dataset(base_dirs, output_dirs, selected_clips=None, colors=None, font_name='Arial', text_size=12):
    """Analyze all tracking files in multiple directories"""
    results = {}

    # Setup plot styling
    setup_plot_style(font_name, text_size)

    # Iterate through each dataset directory
    for base_dir in base_dirs:
        base_path = Path(base_dir)
        dataset_name = base_path.name
        print(f"\nProcessing dataset: {dataset_name}")

        # Get the output directories for this dataset
        dataset_output_dirs = output_dirs.get(dataset_name, output_dirs.get("default", {}))

        # Process both train and test directories
        for dataset_type in ['train', 'test']:
            dataset_path = base_path / dataset_type

            # Skip if directory doesn't exist
            if not dataset_path.exists():
                print(f"Directory {dataset_path} not found, skipping.")
                continue

            # Iterate through all segment directories
            for segment_dir in dataset_path.iterdir():
                if segment_dir.is_dir():
                    segment_name = segment_dir.name

                    # Skip if this clip is not in our selected list
                    if selected_clips and dataset_name in selected_clips:
                        if segment_name not in selected_clips[dataset_name]:
                            print(f"Skipping {segment_name} (not in selected clips)")
                            continue

                    # Path to the gt.txt file for this segment
                    gt_file = segment_dir / "gt" / "gt.txt"

                    # Skip if gt file doesn't exist
                    if not gt_file.exists():
                        print(f"GT file not found at {gt_file}, skipping.")
                        continue

                    # Use segment name as clip name
                    clip_name = f"{dataset_name}_{dataset_type}_{segment_dir.name}"
                    print(f"Processing {clip_name}...")

                    # Load and analyze the data
                    try:
                        df = load_tracking_data(gt_file)
                        results[clip_name] = analyze_clip(df, clip_name, dataset_output_dirs, colors, font_name,
                                                          text_size)
                    except Exception as e:
                        print(f"Error processing {clip_name}: {e}")

    return results


def merge_images_by_category(output_base_dir, dataset_names):
    """Merge images of the same category into a single large image, arranged horizontally."""
    # Create directory for merged images
    merged_dir = os.path.join(output_base_dir, 'merged')
    os.makedirs(merged_dir, exist_ok=True)

    # Define image categories to merge
    categories = {
        'speed_distribution': '*speed_distribution_*.png',
        'speed_trend': '*speed_trend_*.png',
        'angular_velocity_distribution': '*angular_velocity_distribution_*.png',
        'angular_velocity_trend': '*angular_velocity_trend_*.png',
        'direction_distribution': '*direction_distribution_*.png',
        'area_distribution': '*area_distribution_*.png',
        'area_trend': '*area_trend_*.png',
        'target_counts': '*target_counts_*.png',
        'feature_correlation': '*feature_correlation_*.png',
    }

    # Process each category
    for category_name, pattern in categories.items():
        image_paths = []

        # Find all images matching the pattern
        for dataset in dataset_names:
            dataset_dir = os.path.join(output_base_dir, dataset)
            for root, _, _ in os.walk(dataset_dir):
                for file_path in glob.glob(os.path.join(root, pattern)):
                    image_paths.append(file_path)

        if not image_paths:
            continue

        # Merge images
        merge_images(image_paths, os.path.join(merged_dir, f'{category_name}_merged.png'))


def merge_images(image_paths, output_path):
    """Horizontally merge multiple images into a single row."""
    if not image_paths:
        return

    # Open all images
    images = [Image.open(path) for path in image_paths]

    # Arrange horizontally: number of columns equals number of images, one row
    n = len(images)

    # Standardize all image sizes
    target_width = target_height = 500  # Standard image size
    resized_images = [img.resize((target_width, target_height), Image.LANCZOS) for img in images]

    # Create new image - width is the sum of all image widths, height is the height of a single image
    result = Image.new('RGB', (target_width * n, target_height), color='white')

    # Paste images horizontally into the result image
    for i, img in enumerate(resized_images):
        result.paste(img, (i * target_width, 0))

    # Save result
    result.save(output_path)
    print(f"Merged {len(images)} images horizontally into {output_path}")


if __name__ == "__main__":
    # Define all dataset paths to analyze
    dataset_paths = [
        r"C:\Users\vranl\MOT17",
        r"C:\Users\vranl\MOT20",
        r"C:\Users\vranl\CTMCV1",
        r"C:\Users\vranl\DanceTrack",
        r"C:\Users\vranl\BEE24",
        r"C:\Users\vranl\MFT25"
    ]

    # Extract dataset names for creating folder structure
    dataset_names = [Path(path).name for path in dataset_paths]

    # Create output directories organized by dataset
    output_base_dir = 'analysis_results'
    output_dirs = create_output_directories(output_base_dir, dataset_names)

    # Define custom colors with HTML color codes
    custom_colors = {
        'hist': '#87CEFA',    # skyblue in HTML
        'line': '#185ea8',    # navy in HTML
        'trend': '#fe919c',   # crimson in HTML
        'polar': '#2CADD6',   # purple in HTML
        'boxplot': '#F08080'  # lightcoral in HTML
    }

    # Define specific clips to analyze for each dataset
    selected_clips = {
        "MOT17Labels": ["MOT17-05-SDP"],
        "MOT20Labels": ["MOT20-02"],
        "DanceTrackLabels": ["dancetrack0069"],
        "CTMCV1Labels": ["PL1Ut-run05"],
        "MFT25": ["BT-001"],
        "BEE24Labels": ["BEE24-01"],
    }

    # Run the analysis with custom styling
    results = analyze_dataset(
        dataset_paths,
        output_dirs,
        selected_clips=selected_clips,
        colors=custom_colors,
        font_name='Cambria',
        text_size=28
    )

    # Save summary results
    summary_df = pd.DataFrame(results).T
    summary_df.to_csv("tracking_analysis_summary.csv")

    print("Analysis complete. Results saved to CSV and plot images organized by dataset and type.")
    print("Merged images saved to 'analysis_results/merged' directory.")

    merge_images_by_category(output_base_dir, dataset_names)

    print("Analysis finished. Results have been saved to CSV and images organized by dataset and type.")
    print("Merged images have been saved to 'analysis_results/merged' directory.")