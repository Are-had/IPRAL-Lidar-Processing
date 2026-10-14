import numpy as np





def calibrate_step_2(AMB_par,analog_parallel , analog_perpendicular ,Pic_perp_meas_cirrus , Pic_parall_meas_cirrus , depol_mol  , depol_cirrus  , alt , alt_min_mol , alt_max_mol ): 

    index_mol = np.where((alt >= alt_min_mol) & (alt <= alt_max_mol))[0]

    # mol and cirrus parameters

    P_parall_true_mol =np.nanmean(AMB_par[index_mol[0]: index_mol[1]])
    P_parall_meas_mol = np.nanmean(analog_parallel[index_mol[0]: index_mol[1]])
    P_perp_meas_mol = np.nanmean(analog_perpendicular[index_mol[0]: index_mol[1]])

    # cirrus parameters
    P_parall_meas_cirrus = Pic_parall_meas_cirrus
    P_perp_meas_cirrus = Pic_perp_meas_cirrus

    d4 = P_parall_true_mol / P_parall_meas_mol

    d2 = (depol_cirrus*d4*P_parall_meas_cirrus*P_perp_meas_mol - depol_mol*P_parall_true_mol*P_perp_meas_cirrus) / (P_parall_meas_cirrus*P_perp_meas_mol - P_parall_meas_mol*P_perp_meas_cirrus)

    d3 = 0

    d1 = (depol_mol*P_parall_true_mol-d2*P_parall_meas_mol)/P_perp_meas_mol

    return d1, d2, d3, d4







def get_max_values(ATB_par, ATB_per, altitude, alt_min, alt_max):
    """
    Finds max of ATB_par, and retrieves ATB_per and Altitude at that specific location.
    """
    
    # 1. Define the altitude range indices
    
    idx_range = np.where((altitude >= alt_min) & (altitude <= alt_max))[0]
    
    if len(idx_range) == 0:
        print("Attention : Aucune donnée dans cette plage d'altitude !")
        return None, None, None

    # 2. Slice both arrays to the altitude range
    
    if np.ndim(ATB_par) == 1:
            
        ATB_par_sub = ATB_par[ idx_range]
        ATB_per_sub = ATB_per[ idx_range]
            # 3. Find the Max in Parallel
        par_max_values = np.nanmax(ATB_par_sub) 
        max_indices_sub = np.argmax(ATB_par_sub)
        
        per_val_at_max_par = ATB_per_sub[max_indices_sub]
        alt_at_max = altitude[idx_range[max_indices_sub]]


        
    else :

        ATB_par_sub = ATB_par[:, idx_range]
        ATB_per_sub = ATB_per[:, idx_range]

        # 3. Find the Max in Parallel
        # This gives the max value for each profile
        par_max_values = np.nanmax(ATB_par_sub, axis=1) 
        # This gives the relative index (0 to N) inside the subset where the max is
        max_indices_sub = np.argmax(ATB_par_sub, axis=1)

        # 4. Retrieve Perpendicular value at the SAME indices
        # We need to create a range of row numbers [0, 1, 2, ... 177]
        n_profiles = ATB_par.shape[0]
        rows = np.arange(n_profiles)
        
        # This is the "magic" numpy indexing: [rows, columns]
        per_val_at_max_par = ATB_per_sub[rows, max_indices_sub]

        # 5. Retrieve the exact Altitude at these indices
        # We map the relative subset indices back to the global altitude array
        global_indices = idx_range[max_indices_sub]
        alt_at_max = altitude[global_indices]

    return par_max_values, per_val_at_max_par, alt_at_max





def fit_clear_sky(alt, signal, range_list):
    """
    Fits S(z) = a * exp(-b * z) using linear regression on log(S).
    This is much more stable than curve_fit for Lidar data.
    """
    # 1. Create masks for the ranges
    mask1 = (alt >= range_list[0][0]) & (alt <= range_list[0][1])
    mask2 = (alt >= range_list[1][0]) & (alt <= range_list[1][1])
    
    # IMPORTANT: Signal must be strictly positive for Log
    mask = (mask1 | mask2) & (signal > 0) & np.isfinite(signal)
    
    x_fit = alt[mask]
    y_fit = signal[mask]
    
    # Safety check
    if len(x_fit) < 2: # Need at least 2 points for a line
        return np.full_like(alt, np.nan)

    # 2. Linearize: ln(y) = ln(a) - b*z
    # We fit Y = C + m*X
    # Slope m = -b
    # Intercept C = ln(a)
    try:
        # polyfit(x, y, 1) fits a degree 1 polynomial (line)
        slope, intercept = np.polyfit(x_fit, np.log(y_fit), 1)
        
        # Recover a and b
        b = -slope
        a = np.exp(intercept)
        
    except Exception as e:
        print(f"Fit failed: {e}")
        return np.full_like(alt, np.nan)
    
    # 3. Apply the model to the full altitude
    # Model: S(z) = a * exp(-b * z)
    fitted_signal = a * np.exp(-b * alt)
    
    return fitted_signal






def calculate_depolarization_ratio_range(ATB_par, ATB_per, altitude, alt_min, alt_max):
    """
    Calcule le ratio de dépolarisation uniquement sur une plage d'altitude donnée.
    
    Args:
        ATB_par (np.array): Signal parallèle (178, 4000)
        ATB_per (np.array): Signal perpendiculaire (178, 4000)
        altitude (np.array): Vecteur altitude (4000,)
        alt_min (float): Altitude min en mètres (ex: 8000)
        alt_max (float): Altitude max en mètres (ex: 12000)
        
    Returns:
        depol_ratio_full (np.array): Matrice complète (178, 4000) avec NaNs hors zone/masque.
        mask_full (np.array): Masque binaire complet (178, 4000).
    """
    

    idx_range = np.where((altitude >= alt_min) & (altitude <= alt_max))[0]
    
    if len(idx_range) == 0:
        print("Attention : Aucune donnée dans cette plage d'altitude !")
        return np.full_like(ATB_par, np.nan), np.zeros_like(ATB_par)


    ATB_par_sub = ATB_par[:, idx_range]
    ATB_per_sub = ATB_per[:, idx_range]

    layer_max = np.max(ATB_per_sub, axis=1, keepdims=True)
    threshold = 0.1 * layer_max
    print("Threshold for local mask:", np.shape(threshold))

    # 4. Créer le masque local
    layer_mask = ((ATB_par_sub > threshold) & (ATB_par_sub > 0) & (ATB_per_sub > 0)).astype(float)
    
    # 5. Calculer la dépolarisation sur la tranche
    par_masked = ATB_par_sub * layer_mask
    per_masked = ATB_per_sub * layer_mask
    total_signal = par_masked + per_masked
    
    layer_depol = np.full_like(ATB_par_sub, np.nan)
    
    valid_indices = total_signal != 0
    layer_depol[valid_indices] = per_masked[valid_indices] / total_signal[valid_indices]
    
    # 6. Reconstruire la matrice complète (178, 4000)
    
    depol_ratio_full = np.full_like(ATB_par, np.nan)
    mask_full = np.zeros_like(ATB_par) 
    
    # On remplit uniquement les colonnes correspondant à l'altitude choisie
    depol_ratio_full[:, idx_range] = layer_depol
    mask_full[:, idx_range] = layer_mask
    
    return depol_ratio_full, mask_full