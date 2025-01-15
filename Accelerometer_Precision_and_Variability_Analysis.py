import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
mpl.rcParams.update({
    'font.size': 16,         # Global font size
    'axes.titlesize': 16,    # Font size of the axes title
    'axes.labelsize': 16,    # Font size of the x and y labels
    'xtick.labelsize': 16,   # Font size of the x tick labels
    'ytick.labelsize': 16,   # Font size of the y tick labels
    'legend.fontsize': 16,   # Font size of the legend
    'figure.titlesize': 18,  # Font size of the figure title
    'figure.figsize': (6.8, 4.8),  # Adjusted figure size for the extra plot
    'figure.dpi': 500,       # Default DPI
    'legend.loc': 'best',    # Default legend location
    'legend.frameon': True   # Legend frame
})

def load_data(file_name, time_column, accel_column, time_range):
    """Loads and filters acceleration data based on the provided time range."""
    time, accel = np.loadtxt(file_name, usecols=(time_column, accel_column), unpack=True)
    sampling_rate = len(time) / np.ptp(time)
    mask = (time > time_range[0]) & (time < time_range[1])
    time_filtered = time[mask]
    accel_filtered = accel[mask]
    time_filtered -= time_filtered[0]  # Normalize time to start from zero
    return time_filtered, accel_filtered, sampling_rate 

def analyze_fall(time, accel, rate, label):
    """Calculates and prints the mean and standard deviation of acceleration."""
    accel_mean = np.mean(accel)
    accel_std = np.std(accel, ddof=1)
    print(f"{label}: # points {len(time)} | Δt: {np.ptp(time):.2f}s | rate {rate:.1f} Hz | a_p: {accel_mean:.3f} ± {accel_std:.3f} m/s²")
    return accel_mean, accel_std

# Load the data
time_fall_1, accel_fall_1, rate_1 = load_data('fall_vertical_1.txt', 0, 3, (3.1619, 3.5582))
time_fall_2, accel_fall_2, rate_2 = load_data('fall_vertical_2.txt', 0, 3, (3.5062, 3.9830))
time_fall_3, accel_fall_3, rate_3 = load_data('fall_vertical_3.txt', 0, 3, (3.4214, 3.9433))
time_static, accel_static, rate_static = load_data('accel_static.txt', 0, 3, (0, 8.78))

# Analyze each fall
mean_1, std_1 = analyze_fall(time_fall_1, accel_fall_1, rate_1, "Free-Fall 1")
mean_2, std_2 = analyze_fall(time_fall_2, accel_fall_2, rate_2, "Free-Fall 2")
mean_3, std_3 = analyze_fall(time_fall_3, accel_fall_3, rate_3, "Free-Fall 3")
mean_static, std_static = analyze_fall(time_static, accel_static, rate_static, "Static")

# Group and centralize the accelerations
accel_fall_1_zero = accel_fall_1 - mean_1
accel_fall_2_zero = accel_fall_2 - mean_2
accel_fall_3_zero = accel_fall_3 - mean_3
accel_static_zero = accel_static - mean_static

accel_fall_all = np.concatenate((accel_fall_1_zero, accel_fall_2_zero, accel_fall_3_zero))

accel_grouped = np.concatenate((accel_fall_1_zero, accel_fall_2_zero, accel_fall_3_zero, accel_static_zero))
print(f"Grouped Accelerations: # points {len(accel_grouped)} | Standard deviation: {np.std(accel_grouped):.3f} m/s²")

# Plotting the histogram and Gaussian curve
def plot_histogram_with_gaussian(static, fall):
    """Plots a histogram of the data with a Gaussian fit."""
    hist_range = (-0.08, 0.08)
    plt.hist(static, bins=15, range=hist_range, density=True, alpha=0.5, color='b', label='Static')
    plt.hist(fall, bins=15, range=hist_range, color='orange', histtype='step', density=True, lw=3.2, label='Free-Fall')
    
    # Fit Gaussian
    mu = 0  # Given that we're centralizing the data
    sigma = np.std(accel_grouped)
    
    # Generate points for Gaussian curve
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, sigma)
    
    plt.plot(x, p, color='#006400', linewidth=2.1, ls='--', label=f'Gaussian')
    
    plt.xlim(hist_range)
    plt.xlabel(r'$a_{p}(\mathrm{m} \, \mathrm{s}^{-2})$')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig('accelerometer_random_uncertainty_gauss.png', bbox_inches='tight')
    #plt.show()

plot_histogram_with_gaussian(accel_static_zero, accel_fall_all)

