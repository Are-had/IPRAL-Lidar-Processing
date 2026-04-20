
import numpy as np
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter1d 
import matplotlib.pyplot as plt

def conc_calculation (data_file_rs):

    Rd   = 287.              #gas constant for dry air [J/(kg.K)]
    Na = 6.022 * 10 **(23)
    Mair = 28.8*10**-3       # Masse molaire de l'air
    
    
    rs_data = read_nc_file(data_file_rs)
    
    alt= np.array(rs_data["alt"]) #alt max = 35408.438 m 

    P =np.array( rs_data["press"])    #air pressure hpa
    T=np.array(rs_data["temp"])   # temp_raw    K
    ro = (100 *P)/(Rd*T)
    
    conc= ro * (Na /Mair )
    
    T_in_C = T - 273.15

    return [conc , alt , T_in_C]




def read_nc_file(path_file) :
    
    #  """    This function reads a NetCDF (.nc) file and returns a netCDF4.Dataset object."""
    try:
        data = Dataset(path_file, "r")
        # File opened successfully!
        return data
    except FileNotFoundError:
        print(f"Error: File not found at path {path_file}")
    except Exception as e:
        print(f"An error occurred: {e}")
        
        



def calculate_AMB_clear(conc , altitude_rs  ,altitude_ipral ) :

    conc = np.interp(altitude_ipral, altitude_rs, conc)


    RAYLEIGH_CROSS_SECTION = 3.2897988e-31  # [m2 sr-1]
    beta_ray_355 = conc * RAYLEIGH_CROSS_SECTION
    alpha_ray_355 = beta_ray_355 / 0.119 
    n=conc.shape[0]
    
    AMB_clear = np.full(n, np.nan)

    dz = (np.diff(altitude_ipral))
    trapz_coef = ((alpha_ray_355[:-1] + alpha_ray_355[ 1:]) / 2) * dz
    sum1 = np.insert( np.nancumsum(trapz_coef) ,0,0 )

    AMB_clear = beta_ray_355 * np.exp(-2.0 * sum1)   
    
     
    return AMB_clear  , beta_ray_355 , alpha_ray_355






def remove_nans_interpolation(signal , altitude, N_valid=10):

    signal_clean = np.nan_to_num(signal, nan=0.0)
    mask_valid = ~np.isnan(signal)
    if np.sum(mask_valid) > N_valid:  # Au moins N_valid points valides
        signal_clean = np.interp(altitude, altitude[mask_valid], signal[mask_valid])
    else:
        signal_clean = np.nan_to_num(signal, nan=1e-7)
    return signal_clean









def gaussian_filter(signal, altitude, window_size=500, min_sigma=2, max_sigma=20, max_altitude=10000):
    """
    Apply Gaussian filter to signal(s).
    
    Parameters:
    -----------
    signal : array-like
        1D array (n_altitude,) for single profile
        OR 2D array (n_profiles, n_altitude) for multiple profiles
    altitude : array-like
        1D array of altitudes (n_altitude,)
    
    Returns:
    --------
    filtered : array with same shape as input signal
    """
    # Convert to numpy array
    signal = np.asarray(signal)
    altitude = np.asarray(altitude)
    
    # Check if input is 1D or 2D
    if signal.ndim == 1:
        return _filter_single_profile(signal, altitude, window_size, min_sigma, max_sigma, max_altitude)
    
    elif signal.ndim == 2:
        n_profiles, n_altitude = signal.shape
        filtered = np.zeros_like(signal)
        
        for i in range(n_profiles):
            filtered[i, :] = _filter_single_profile(
                signal[i, :], altitude, window_size, min_sigma, max_sigma, max_altitude
            )
        
        return filtered
    
    else:
        raise ValueError(f"Signal must be 1D or 2D, got {signal.ndim}D")


def _filter_single_profile(signal, altitude, window_size, min_sigma, max_sigma, max_altitude):
    """Helper function to filter a single 1D profile"""
    n_points = len(signal)
    filtered = np.zeros_like(signal)
    
    for i in range(n_points):
        half_window = window_size // 2
        start = max(0, i - half_window)
        end = min(n_points, i + half_window)
        
        window_signal = signal[start:end]
        sigma = _get_sigma(altitude[i], min_sigma, max_sigma, max_altitude)
        filtered_window = gaussian_filter1d(window_signal, sigma=sigma)
        
        # Adjust index to get the value corresponding to the current point
        filtered[i] = filtered_window[i - start]
    
    return filtered



def _get_sigma(altitude, min_sigma=5, max_sigma=30, max_altitude=60000):
    # Linearly increase sigma with altitude
    slope = (max_sigma - min_sigma) / max_altitude
    return min_sigma + slope * altitude









def Calibration(rcs , AMB_clear , altitude , calibration_mask1=None , calibration_mask2=None ,N =60 , visual = True , seed=True) :
    
    if (seed) :
    
        np.random.seed(5)  
    

    mask_1 = (altitude >= calibration_mask1[0]) & (altitude <= calibration_mask1[-1])
    indices_1 = np.where(mask_1)[0]
    selected_idx_1 = np.random.choice(indices_1, size=N, replace=True)
    
    

    mask_2 = (altitude >= calibration_mask2[0]) & (altitude <= calibration_mask2[-1])
    indices_2 = np.where(mask_2)[0]
    selected_idx_2 = np.random.choice(indices_2, size=N, replace=True)

    S1= rcs[selected_idx_1]
    S2=rcs[selected_idx_2]

    AMB_1=AMB_clear[selected_idx_1]
    AMB_2=AMB_clear[selected_idx_2]

    K = (AMB_1-AMB_2)/(S1-S2)
    delta= (S2*AMB_1 - S1*AMB_2)/(AMB_1-AMB_2)


    K_avrg = np.nanmean(K)
    delta_avrg = np.nanmean(delta)




    if(visual) :

        
        # --- Plot histogram of K ---

        plt.figure(figsize=(8, 5))
        plt.hist(K, bins='auto', edgecolor='black', alpha=0.7 , label =f"the avrg of K is : {K_avrg}")
        plt.xlabel('K values')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of K (n={N})')
        plt.grid(True)
        plt.legend()
        plt.show()
        
        
        # --- Plot histogram of delta  ---
        
        plt.figure(figsize=(8, 5))
        plt.hist(delta, bins='auto', edgecolor='black', alpha=0.7 , label =f"the avrg of delta is :{delta_avrg}")
        plt.xlabel('delta values')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of delta (n={N})')
        plt.grid(True)
        plt.legend()
        plt.show()





    return K_avrg , delta_avrg








def cost_function(params, rcs, amb_clear, correction_ranges, altitude):
    """
    correction_ranges: list of tuples [(min1, max1), (min2, max2), ...]
    """
    K_init, delta_init = params 
    
    # Créer un mask qui combine tous les ranges
    combined_mask = np.zeros_like(altitude, dtype=bool)
    
    for range_min, range_max in correction_ranges:
        mask = (altitude >= range_min) & (altitude <= range_max)
        combined_mask |= mask  # OR logique pour combiner les masks
    
    # Appliquer le mask combiné
    atb_masked = K_init * (rcs[combined_mask] - delta_init)
    amb_clear_masked = amb_clear[combined_mask]
    
    mape = np.nanmean(np.abs((atb_masked - amb_clear_masked) / amb_clear_masked)) * 100
    return mape


def optimize(rcs, AMB_clear, altitude, comparison_ranges, K_init, delta_init, method='Nelder-Mead'):
    """
    comparison_ranges: list of tuples for multiple ranges
    """
    from scipy.optimize import minimize
    
    initial_params = [K_init, delta_init]
    
    result = minimize(
        cost_function,
        initial_params, 
        args=(rcs, AMB_clear, comparison_ranges, altitude),
        method=method,
        options={'maxiter': 1000}
    )
    
    K_opt, delta_opt = result.x
    
    return K_opt, delta_opt, result










def merged_signal_hanning(analog, photocounting, altitude, transition_start=15000, transition_end=20000):
    """
    Merge analog and photocounting signals with Hanning window transition.
    
    Parameters:
    -----------
    analog : array-like
        1D array (n_altitude,) for single profile
        OR 2D array (n_profiles, n_altitude) for multiple profiles
    photocounting : array-like
        Same shape as analog
    altitude : array-like
        1D array of altitudes (n_altitude,)
    
    Returns:
    --------
    merged_signal : array with same shape as input
    """
    # Convert to numpy arrays
    analog = np.asarray(analog)
    photocounting = np.asarray(photocounting)
    altitude = np.asarray(altitude)
    
    # Check if input is 1D or 2D
    if analog.ndim == 1:
        # Single profile
        return _merge_single_profile(analog, photocounting, altitude, transition_start, transition_end)
    
    elif analog.ndim == 2:
        # Multiple profiles - merge each one
        n_profiles, n_altitude = analog.shape
        merged_signal = np.zeros_like(analog)
        
        for i in range(n_profiles):
            merged_signal[i, :] = _merge_single_profile(
                analog[i, :], photocounting[i, :], altitude, transition_start, transition_end
            )
        
        return merged_signal
    
    else:
        raise ValueError(f"Signal must be 1D or 2D, got {analog.ndim}D")


def _merge_single_profile(analog, photocounting, altitude, transition_start, transition_end):
    """Helper function to merge a single 1D profile"""
    merged_signal = np.zeros_like(analog)
    
    mask_low = altitude < transition_start
    mask_high = altitude > transition_end
    mask_transition = (altitude >= transition_start) & (altitude <= transition_end)
    
    merged_signal[mask_low] = analog[mask_low]
    merged_signal[mask_high] = photocounting[mask_high]
    
    # Zone de transition avec fenêtre de Hanning
    if np.any(mask_transition):
        n_trans = np.sum(mask_transition)
        hanning_window = 0.5 * (1 - np.cos(np.pi * np.arange(n_trans) / (n_trans - 1)))
        
        merged_signal[mask_transition] = (
            (1 - hanning_window) * analog[mask_transition] + 
            hanning_window * photocounting[mask_transition]
        )
    
    return merged_signal