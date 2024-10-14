import numpy as np
from typing import List, Dict, Tuple, NamedTuple
import json
from datetime import datetime
import csv
from scipy.interpolate import Akima1DInterpolator
from scipy import stats

class ModelStats(NamedTuple):
    r_squared: float
    normalized_chi_square: float
    residual_mean: float
    residual_std: float
    rmse: float
    aic: float
    normality_p_value: float
    params: np.ndarray
    param_errors: np.ndarray

def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the R-squared (coefficient of determination) value."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def normalized_chi_square(y_true: np.ndarray, y_pred: np.ndarray, sigma: np.ndarray, num_params: int) -> float:
    """Calculate the normalized chi-square value."""
    chi_square = np.sum(((y_true - y_pred) / sigma) ** 2)
    dof = len(y_true) - num_params
    return chi_square / dof

def residual_stats(residuals: np.ndarray) -> Tuple[float, float]:
    """Calculate mean and standard deviation of residuals."""
    return np.mean(residuals), np.std(residuals)

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Square Error (RMSE)."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def aic(y_true: np.ndarray, y_pred: np.ndarray, num_params: int) -> float:
    """Calculate Akaike Information Criterion (AIC)."""
    n = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)
    return 2 * num_params + n * np.log(rss / n)

def compute_model_stats(data: np.ndarray, fit: np.ndarray, sigma: np.ndarray, num_params: int, params: np.ndarray, pcov: np.ndarray) -> ModelStats:
    """Compute all statistics for a given dataset and its fit."""
    residuals = data - fit
    r_sq = r_squared(data, fit)
    norm_chi_sq = normalized_chi_square(data, fit, sigma, num_params)
    res_mean, res_std = residual_stats(residuals)
    rmse_val = rmse(data, fit)
    aic_val = aic(data, fit, num_params)
    _, p_value = stats.normaltest(residuals)
    param_errors = np.sqrt(np.diag(pcov))
    
    return ModelStats(
        r_squared=r_sq,
        normalized_chi_square=norm_chi_sq,
        residual_mean=res_mean,
        residual_std=res_std,
        rmse=rmse_val,
        aic=aic_val,
        normality_p_value=p_value,
        params=params,
        param_errors=param_errors
    )

def analyze_datasets(time_data: List[np.ndarray], az_data: List[np.ndarray], 
                     fit_data: List[np.ndarray], sigma_data: List[np.ndarray], 
                     num_params: int, model_name: str, popt_data: List[np.ndarray], 
                     pcov_data: List[np.ndarray]) -> List[Dict[str, ModelStats]]:
    """
    Analyze multiple datasets for a given model.
    
    Parameters:
    time_data (List[np.ndarray]): List of time arrays for each dataset
    az_data (List[np.ndarray]): List of acceleration data arrays for each dataset
    fit_data (List[np.ndarray]): List of fitted data arrays for each dataset
    sigma_data (List[np.ndarray]): List of sigma (uncertainty) arrays for each dataset
    num_params (int): Number of parameters in the model
    model_name (str): Name of the model (e.g., 'Turbulent', 'Generalized')
    popt_data (List[np.ndarray]): List of optimal parameter arrays for each dataset
    pcov_data (List[np.ndarray]): List of covariance matrices for each dataset
    
    Returns:
    List[Dict[str, ModelStats]]: List of dictionaries containing stats for each dataset
    """
    results = []
    for i, (time, data, fit, sigma, popt, pcov) in enumerate(zip(time_data, az_data, fit_data, sigma_data, popt_data, pcov_data)):
        stats = compute_model_stats(data, fit, sigma, num_params, popt, pcov)
        results.append({f"{model_name} Model - Dataset {i+1}": stats})
    return results

def save_results_to_csv(results: List[Dict[str, ModelStats]], filename: str):
    """
    Save the analysis results to a CSV file.
    
    Parameters:
    results (List[Dict[str, ModelStats]]): List of dictionaries containing stats for each dataset
    filename (str): Name of the CSV file to save
    """
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['R-squared', 'Chi-square', 
                      'Residual Mean', 'RMSE', 'AIC', 'Normality p-value', 
                      'v_0', 'v_0_error', 'v_t', 'v_t_error', 'offset', 'offset_error']
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        fall_turb_offset_index = 0
        for result_dict in results:
            for model_dataset, stats in result_dict.items():
                model_parts = model_dataset.split(' Model - Dataset ')
                if len(model_parts) == 2:
                    dataset_type, dataset_number = model_parts
                else:
                    dataset_type, dataset_number = model_dataset, "Unknown"
                
                row = {
                    'R-squared': f"{stats.r_squared:.5f}",
                    'Chi-square': f"{stats.normalized_chi_square:.3f}",
                    'Residual Mean': f"{stats.residual_mean:.6f}",
                    'RMSE': f"{stats.rmse:.4f}",
                    'AIC': f"{stats.aic:.1f}",
                    'Normality p-value': f"{stats.normality_p_value:.3f}"
                }
                
                def format_value(value):
                    return f"{value:.12f}" if value is not None else ""

                row['v_0'] = format_value(stats.params[0])
                row['v_0_error'] = format_value(stats.param_errors[0])
                row['v_t'] = format_value(stats.params[1])
                row['v_t_error'] = format_value(stats.param_errors[1])
                row['offset'] = format_value(stats.params[2])
                row['offset_error'] = format_value(stats.param_errors[2])
                
                writer.writerow(row)
    
    print(f"Results saved to {filename}")

def generate_grouped_std(time_list, az_list, dataset_name, acquisition_frequency=200):
    """
    Gera os dados médios e o desvio padrão a partir de listas de tempo e aceleração.
    
    Parâmetros:
    - time_list: Lista de arrays de tempo.
    - az_list: Lista de arrays de aceleração correspondentes.
    - acquisition_frequency: Frequência de aquisição do acelerômetro (padrão: 200 Hz).
    
    Retorno:
    - common_time: Array de tempo comum.
    - mean_data: Array dos valores médios de aceleração.
    - std_data: Array dos desvios padrão de aceleração.
    """
    # Calcular a duração total com base nos limites comuns dos tempos
    min_time = max([time.min() for time in time_list])
    max_time = min([time.max() for time in time_list])
    total_duration = max_time - min_time
    
    # Estimar o número de pontos baseado na frequência de aquisição e duração
    num_points = int(total_duration * acquisition_frequency)
    
    # Definir um intervalo de tempo comum baseado no mínimo e máximo
    common_time = np.linspace(min_time, max_time, num_points)

    # Interpolar os dados para o tempo comum
    interpolated_data = []
    for time, az in zip(time_list, az_list):
        interpolator = Akima1DInterpolator(time, az)
        interpolated_data.append(interpolator(common_time))

    # Converter para array numpy para facilitar a manipulação
    interpolated_data = np.array(interpolated_data)

    # Calcular média e desvio padrão
    mean_data = np.mean(interpolated_data, axis=0)
    std_data = np.std(interpolated_data, axis=0)

    data = np.column_stack((common_time, mean_data, std_data))
    np.savetxt(dataset_name+'_Grouped_Data.txt', data)

    return [common_time], [mean_data], [std_data]
