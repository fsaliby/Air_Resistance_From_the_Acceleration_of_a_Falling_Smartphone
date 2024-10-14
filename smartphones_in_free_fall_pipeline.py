from data_utils import load_pickle, zero_time, calculate_statistics, display_statistics, define_sigma
from typing import List
import numpy as np
import turbulent
from stats import analyze_datasets, save_results_to_csv, generate_grouped_std

def run_pipeline() -> None:
    
    # Load the data
    data = load_pickle('clean_data.pkl')

    # Access datasets
    fall_time: List[np.ndarray] = data['fall_time']
    fall_az: List[np.ndarray] = data['fall_az']
    up_down_time: List[np.ndarray] = data['up_down_time']
    up_down_az: List[np.ndarray] = data['up_down_az']

    # Ensure times are zeroed
    fall_time = zero_time(fall_time)
    up_down_time = zero_time(up_down_time)

    # Calculate data basic statistics
    fall_stats = calculate_statistics(fall_time, fall_az)
    up_down_stats = calculate_statistics(up_down_time, up_down_az)

    # Define sigma for all datasets
    sigma_fall, sigma_up_down = define_sigma(fall_az, up_down_az)

    # Fit turbulent model to all datasets
    fit_fall_turb = turbulent.fit(fall_time, fall_az, sigma_fall, dataset_name='Fall')
    fit_up_down_turb = turbulent.fit(up_down_time, up_down_az, sigma_up_down, dataset_name='Up-Down')
    
    fall_turb_results = analyze_datasets(fall_time, fall_az, fit_fall_turb.a_fit, sigma_fall, 
                                         num_params=3, model_name="Fall Turbulent", 
                                         popt_data=fit_fall_turb.popt, pcov_data=fit_fall_turb.pcov)
    
    up_down_turb_results = analyze_datasets(up_down_time, up_down_az, fit_up_down_turb.a_fit, sigma_up_down, 
                                            num_params=3, model_name="Up-Down Turbulent", 
                                            popt_data=fit_up_down_turb.popt, pcov_data=fit_up_down_turb.pcov)
    
    save_results_to_csv(fall_turb_results, "Fall_individual_results.csv")
    save_results_to_csv(up_down_turb_results, "Up_Down_individual_results.csv")

    # Generate the grouped and standard deviation data for Fall and Up-Down
    common_time_fall, grouped_fall, std_fall = generate_grouped_std(fall_time, fall_az, 'Fall')
    common_time_up_down, grouped_up_down, std_up_down = generate_grouped_std(up_down_time, up_down_az, 'Up-Down')
    
    # Fit turbulent model to grouped value of all datasets
    fit_fall_grouped = turbulent.fit(common_time_fall, grouped_fall, std_fall, dataset_name='Fall')
    fit_up_down_grouped = turbulent.fit(common_time_up_down, grouped_up_down, std_up_down, dataset_name='Up-Down')
    
    fall_grouped_results = analyze_datasets(common_time_fall, grouped_fall, fit_fall_grouped.a_fit, std_fall,
                                         num_params=3, model_name="Fall Mean",
                                         popt_data=fit_fall_grouped.popt, pcov_data=fit_fall_grouped.pcov)

    up_down_grouped_results = analyze_datasets(common_time_up_down, grouped_up_down, fit_up_down_grouped.a_fit, std_up_down,
                                            num_params=3, model_name="Up-Down Mean",
                                            popt_data=fit_up_down_grouped.popt, pcov_data=fit_up_down_grouped.pcov)

    save_results_to_csv(fall_grouped_results, "Fall_grouped_results.csv")
    save_results_to_csv(up_down_grouped_results, "Up_Down_grouped_results.csv")

if __name__ == "__main__":
    run_pipeline()
