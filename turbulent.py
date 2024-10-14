from typing import NamedTuple, List
import numpy as np
from scipy.optimize import curve_fit
from constants import GRAVITY

class FitResult(NamedTuple):
    popt: np.ndarray
    pcov: np.ndarray
    a_fit: np.ndarray
    residuals: np.ndarray

def validate_inputs(t: np.ndarray, v_0: float, v_t: float):
    """
    Validate the input parameters for the compute_velocity function.

    Parameters:
    t    : np.ndarray
        Time array starting from zero.
    v_0  : float
        Initial velocity (m/s). Positive is upward, negative is downward.
    v_t  : float
        Terminal velocity magnitude (m/s), must be positive.

    Raises:
    ValueError: If any of the input conditions are not met.
    """
    if t.size == 0:
        raise ValueError("Time array is empty. Please provide a non-empty time array starting from zero.")

    if t[0] != 0:
        raise ValueError("Time array must start from zero.")

    if not np.all(np.diff(t) >= 0):
        raise ValueError("Time array must be monotonically increasing.")

    if np.isnan(v_0) or np.isinf(v_0):
        raise ValueError("Initial velocity v_0 must be a finite number.")

    if v_t <= 0 or np.isnan(v_t) or np.isinf(v_t):
        raise ValueError("Terminal velocity v_t must be a positive, finite number.")

    if np.isnan(t).any() or np.isinf(t).any():
        raise ValueError("Time array must contain finite numbers.")

def velocity(t: np.ndarray, v_0: float, v_t: float) -> np.ndarray:
    """
    Compute the velocity v(t) under turbulent drag for different initial velocities.

    Parameters:
    t    : np.ndarray
        Time array starting from zero.
    v_0  : float
        Initial velocity (m/s). Positive is upward, negative is downward.
    v_t  : float
        Terminal velocity magnitude (m/s), must be positive.

    Returns:
    v    : np.ndarray
        Velocity array corresponding to the time array t.
    """
    # Validate inputs
    validate_inputs(t, v_0, v_t)

    v = np.zeros_like(t)

    if v_0 >= 0:
        # Case 1: Initially upward motion
        arctan_v0_vt = np.arctan(v_0 / v_t)
        t_apex = (v_t / GRAVITY) * arctan_v0_vt
        upward_mask = t <= t_apex

        # Upward motion using the tan function
        v[upward_mask] = v_t * np.tan(arctan_v0_vt - (GRAVITY * t[upward_mask]) / v_t)

        # Downward motion after reaching the apex
        if np.any(~upward_mask):
            downward_time = t[~upward_mask] - t_apex
            v[~upward_mask] = -v_t * np.tanh((GRAVITY * downward_time) / v_t)
    else:
        if v_0 == -v_t:
            # Case 2a: Constant terminal velocity
            v = -v_t * np.ones_like(t)
        elif v_0 < -v_t:
            # Case 2b: Faster than terminal velocity (use coth)
            def arccoth(x):
                return 0.5 * np.log((x + 1) / (x - 1))

            def coth(x):
                return 1 / np.tanh(x)

            argument = arccoth(v_0 / (-v_t)) + (GRAVITY * t) / v_t
            v = -v_t * coth(argument)
        else:
            # Case 2c: Slower than terminal velocity (use tanh)
            argument = np.arctanh(v_0 / (-v_t)) + (GRAVITY * t) / v_t
            v = -v_t * np.tanh(argument)

    return v

def proper_acceleration(t: np.ndarray, v_0: float, v_t: float, offset: float) -> np.ndarray:
    """Calculates the proper acceleration for an object subject to turbulent resistance."""
    v = velocity(t, v_0, v_t)
    ap = -GRAVITY * np.sign(v) * np.abs(v / v_t)**2  # Proper acceleration for turbulent resistance
    return ap + offset

def fit(time_data: List[np.ndarray], az_data: List[np.ndarray], sigma_data: List[np.ndarray], dataset_name: str) -> FitResult:
    """
    Fits the turbulent resistance model to all experimental datasets.

    Parameters:
    time_data (List[np.ndarray]): List of time arrays for the datasets.
    az_data (List[np.ndarray]): List of acceleration arrays for the datasets.
    sigma_data (List[np.ndarray]): List of sigma (uncertainty) arrays for the datasets.
    dataset_name (str): The name of the dataset ('fall' or 'up and down').

    Returns:
    FitResult: Named tuple containing lists of fitted parameters, covariance matrices,
               fitted acceleration values, and residuals for each dataset.

    Raises:
    ValueError: If an unknown dataset name is provided.
    """
    
    if dataset_name == 'Fall':
        initial_guess = [-0.5, 13, -0.05]  # (v_0, v_t, offset)
    elif dataset_name == 'Up-Down':
        initial_guess = [4, 13, -0.1]  # (v_0, v_t, offset)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    all_popt, all_pcov, all_a_fit, all_res = [], [], [], []

    for t, a, sigma in zip(time_data, az_data, sigma_data):
        try:
            popt, pcov = curve_fit(
                proper_acceleration, 
                t, 
                a, 
                sigma=sigma, 
                absolute_sigma=True, 
                p0=initial_guess, 
                bounds=([-20, 0, -1], [20, 100, 1]), 
                max_nfev=10_000
            )
            
            a_fit = proper_acceleration(t, *popt)
            res = a - a_fit
            
            all_popt.append(popt)
            all_pcov.append(pcov)
            all_a_fit.append(a_fit)
            all_res.append(res)
        except RuntimeError as e:
            print(f"Curve fit failed for a dataset: {e}")
            # Append None or placeholder values if the fit fails
            all_popt.append(None)
            all_pcov.append(None)
            all_a_fit.append(None)
            all_res.append(None)
    
    return FitResult(popt=all_popt, pcov=all_pcov, a_fit=all_a_fit, residuals=all_res)
