from datetime import datetime, timedelta
import numpy as np
from netCDF4 import Dataset
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates




def days_since_1970_to_datetime(days_array):
    epoch = datetime(1970, 1, 1)

    return [epoch + timedelta(days=float(d)) for d in days_array]  




def get_corrected_signal(data_sirta  , alt   ,rcs_="rcs_12") :
  
    rcs = np.array(data_sirta[rcs_][:])
    back_rcs = np.array(data_sirta[f"bckgrd_{rcs_}"][:])
    rcs_rc = substract_bckgrd(rcs ,back_rcs , alt**2)
    
    return np.array(rcs_rc) 



def substract_bckgrd(rcs, bckgrd, r_square):
   
    data = ((rcs / r_square).T - bckgrd).T * r_square

    return np.array(data)






def read_nc_file_(path_file):
    """Retourne un dictionnaire comme H5"""
    
    if not os.path.exists(path_file) :
        print(" File does not exist !")
        return None 
    
    
    try:
        nc_data = Dataset(path_file, "r")
        # Convertir en dictionnaire 
        data_dict = {}
        for var_name in nc_data.variables:
            data_dict[var_name] = nc_data.variables[var_name][:]
        return data_dict
    except Exception as e:
        print(f"An error occurred: {e}")
        return None







def plot_rcs(rcs , time , alt , title , vmax=None , vmin=0 , y_limit=(0,14000), save=False) :

    plt.set_cmap("jet")
    plt.title(title)
    plt.pcolormesh(time , alt ,(rcs.T) , vmin = vmin , vmax=vmax )
    plt.colorbar(label="rcs ")
    plt.ylim(y_limit)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xlabel("Time (HH:MM)")
    plt.ylabel("Altitude (m)")
    if save :
        plt.savefig(f"figures/{title}.png", dpi=300, bbox_inches='tight')
  



def get_indx_from_range_time_sirta( start_time, end_time , time ):

    start_time = np.datetime64(start_time)
    end_time = np.datetime64(end_time)
    
    indx_range = np.where((time >= start_time) & (time <= end_time))[0]
    
    return indx_range
