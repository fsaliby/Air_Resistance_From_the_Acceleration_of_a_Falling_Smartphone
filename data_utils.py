import pickle
import numpy as np
from typing import Dict, Any, List, NamedTuple

class DatasetStats(NamedTuple):
    num_points: int
    time_min: float
    time_max: float
    time_range: float
    az_min: float
    az_max: float
    az_range: float

def load_pickle(file_path: str) -> Dict[str, Any]:
    """Load data from a pickle file."""
    try:
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)
        return data_dict
    except (FileNotFoundError, IOError, pickle.PickleError) as e:
        print(f"Error loading pickle file: {e}")
        return {}

def save_pickle(data: Dict[str, Any], file_path: str) -> None:
    """Save data to a pickle file."""
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    except (IOError, pickle.PickleError) as e:
        print(f"Error saving pickle file: {e}")

def zero_time(time_arrays: List[np.ndarray]) -> List[np.ndarray]:
    """Ensure all time arrays start from zero."""
    return [time - time[0] for time in time_arrays]

def calculate_statistics(time_data: List[np.ndarray], az_data: List[np.ndarray]) -> List[DatasetStats]:
    """Calculate basic statistics for time and acceleration data."""
    stats = []
    
    for time, az in zip(time_data, az_data):
        stats.append(DatasetStats(
            num_points=len(time),
            time_min=np.min(time),
            time_max=np.max(time),
            time_range=np.ptp(time),
            az_min=np.min(az),
            az_max=np.max(az),
            az_range=np.ptp(az)
        ))
    
    return stats

def display_statistics(stats: List[DatasetStats], dataset_name: str) -> None:
    """Display the statistics for the datasets."""
    total_datasets = len(stats)
    print(f"\n=== {dataset_name} Statistics ===")
    print(f"Total number of datasets: {total_datasets}")
    
    time_min_all = min(s.time_min for s in stats)
    time_max_all = max(s.time_max for s in stats)
    az_min_all = min(s.az_min for s in stats)
    az_max_all = max(s.az_max for s in stats)
    
    print(f"Overall time range across all datasets: {time_min_all:.2f} to {time_max_all:.2f} seconds")
    print(f"Overall acceleration range across all datasets: {az_min_all:.2f} to {az_max_all:.2f} m/s^2")
    
    for i, stat in enumerate(stats, 1):
        print(f"\nDataset {i}:")
        print(f"  Number of points: {stat.num_points}")
        print(f"  Time range: {stat.time_min:.2f} to {stat.time_max:.2f} seconds (Range: {stat.time_range:.2f})")
        print(f"  Acceleration range: {stat.az_min:.2f} to {stat.az_max:.2f} m/s^2 (Range: {stat.az_range:.2f})")

def define_sigma(fall_az: list, up_down_az: list) -> tuple:
    """
    Define sigma arrays for fall and up_down acceleration data with an optional user input for sigma.
    Sigma represents the uncertainty in the acceleration measurements.
    If the user doesn't input a value or enters an invalid value, the default value of 0.022 is used.
    Only positive values are accepted.

    Parameters:
    fall_az (list of np.ndarray): List of acceleration arrays for fall datasets.
    up_down_az (list of np.ndarray): List of acceleration arrays for up and down datasets.

    Returns:
    tuple: Two lists of sigma arrays, one for fall and one for up and down datasets.
    """
    default_sigma = 0.022

    while True:
        user_input = input(f"Please enter a sigma value (default is {default_sigma}): ")

        if user_input == "":
            input_sigma = default_sigma
            print(f"Using default sigma: {input_sigma}")
            break
        else:
            try:
                input_sigma = float(user_input)
                if input_sigma > 0:
                    print(f"Using user-defined sigma: {input_sigma}")
                    break
                else:
                    print("Sigma must be a positive number. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a valid positive number.")

    sigma_fall = [np.full_like(az, input_sigma) for az in fall_az]
    sigma_up_down = [np.full_like(az, input_sigma) for az in up_down_az]

    return sigma_fall, sigma_up_down
