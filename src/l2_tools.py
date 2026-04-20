import numpy as np


def klett_backscatter_aerosol_simplifié(S, l_aer, beta_mol, index_reference, reference_range, beta_aer_R0, bin_length, l_mol=8.73965404, affiche=False):

    # calculat B : ----------------------------
    beta_mol_R0, S_R0 = get_reference_values(beta_mol, index_reference, S, reference_range)
    B = S_R0 / (beta_aer_R0 + beta_mol_R0)
    
    # calculat A : -------------------------------
    tau_integral_argument = (l_aer - l_mol) * beta_mol
    tau_integral = integrate_from_reference_trapezoid(tau_integral_argument, index_reference , bin_length)
    
    
    tau = np.exp(-2 * tau_integral)
    A = S * tau
    

    
    # calculat C : ------------------------------------
    C_integral_argument = l_aer * S * tau
    C_integral = integrate_from_reference_trapezoid(C_integral_argument, index_reference , bin_length)
    C = 2 * C_integral
    
    # Sum of aerosol and molecular backscatter coefficients.
    beta_sum = A / (B - C)
    
    
    # Aerosol backscatter coefficient.
    beta_aerosol = beta_sum - beta_mol
    
    if (affiche):
        relative_error = (beta_sum - beta_mol) / beta_mol * 100.0
        print('Mean relative error: ', np.mean(relative_error), '%')
        print('Max relative error: ', np.max(np.abs(relative_error)), '%')
        print('RMS relative error: ', np.sqrt(np.mean(relative_error**2)), '%')


    return np.array(beta_aerosol), np.array(beta_sum)




def get_reference_values(beta_molecular, index_reference, range_corrected_signal, reference_range):
    idx_min = index_reference - reference_range
    idx_max = index_reference + reference_range
    
    range_corrected_signal_reference = np.mean(range_corrected_signal[idx_min:idx_max+1])
    beta_molecular_reference = np.mean(beta_molecular[idx_min:idx_max+1]) 
    
    return beta_molecular_reference, range_corrected_signal_reference
    
    


def integrate_from_reference_trapezoid(integral_argument, index_reference, bin_length):
    """
    Calculate the cumulative integral of `integral_argument` from the reference point
    using trapezoidal integration.
    
    Parameters
    ----------
    integral_argument : array_like
        The argument to integrate (e.g., LR_part * RCS * exp(2*S_m))
    index_reference : integer
        The index of the reference height (bins)
    bin_length : float
        The vertical bin length (m)
    
    Returns
    -------
    integral : array_like
        The cumulative integral from the reference point
    """
    N_Z = len(integral_argument)
    integral = np.zeros(N_Z)
    
    # Set reference point to zero
    integral[index_reference] = 0.0
    
    # Below reference: integrate from ref down to beginning
    for i_Z in range(index_reference - 1, -1, -1):
        # Trapezoidal rule: (f[i+1] + f[i]) / 2 * dz
        integral[i_Z] = integral[i_Z + 1] - 0.5 * (integral_argument[i_Z + 1] + 
                                                     integral_argument[i_Z]) * bin_length
    
    # Above reference: integrate from ref up to end
    for i_Z in range(index_reference + 1, N_Z):
        # Trapezoidal rule: note the sign change
        integral[i_Z] = integral[i_Z - 1] + 0.5 * (integral_argument[i_Z - 1] + 
                                                     integral_argument[i_Z]) * bin_length
    
    return integral








def L1_2_L2(ATB_par  , INDEX_FOR_THE_CALIBRATION , alt_sirta , beta_ray , LR =17
            , reference_range = 50 , beta_aerosol_reference = 1e-9 ):


    index_reference = np.argmin(np.abs(alt_sirta -INDEX_FOR_THE_CALIBRATION ))
    bin_length = np.abs(np.median(np.diff(alt_sirta)))

    print(f"calibration range {INDEX_FOR_THE_CALIBRATION -reference_range * bin_length } m  to {INDEX_FOR_THE_CALIBRATION + reference_range * bin_length } m")


    beta_aerosol, beta_sum = klett_backscatter_aerosol_simplifié( ATB_par,LR,beta_ray,index_reference,reference_range,beta_aerosol_reference,bin_length,8*np.pi/3 , affiche=False)

    return beta_aerosol , beta_sum








def filter_negative_backscatter(beta_parallel, beta_perpendicular):
    """
    Filtre les valeurs négatives des composantes parallèle et perpendiculaire,
    puis recalcule le beta total propre.
    
    Args:
        beta_parallel (np.array): Backscatter particulaire parallèle
        beta_perpendicular (np.array): Backscatter particulaire perpendiculaire
        
    Returns:
        beta_total_clean: Somme des composantes nettoyées
        beta_par_clean: Composante parallèle nettoyée (>0)
        beta_per_clean: Composante perpendiculaire nettoyée (>0)
        mask_par: Masque binaire utilisé pour le parallèle
        mask_per: Masque binaire utilisé pour le perpendiculaire
    """
    
    # 1. Création des Masques (1 si > 0, sinon 0)
    # On utilise np.where pour dire : "Là où c'est > 0 met 1, sinon met 0"
    mask_par = np.where(beta_parallel > 0, 1, 0)
    mask_per = np.where(beta_perpendicular > 0, 1, 0)
    
    # 2. Application des Masques (Multiplication)
    # Les valeurs négatives deviennent 0 (car multipliées par 0)
    # Les valeurs positives restent identiques (car multipliées par 1)
    beta_par_clean = beta_parallel * mask_par
    beta_per_clean = beta_perpendicular * mask_per
    
    # 3. Calcul du Total Propre
    beta_total_clean = beta_par_clean + beta_per_clean
    
    return beta_total_clean,  mask_par, mask_per
