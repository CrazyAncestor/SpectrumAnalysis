import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
from .statistics import avg_std_of_time_freq_domain
from .io import delete_hdu_from_fits

def find_B_field_idx(fits_file, hdu_id, B_field=None):
    """
    Find the index of the specified B-field in the list of B-fields.
    
    """
    with fits.open(fits_file) as hdul:
        header = hdul[hdu_id].header
        b_field_list = [header[f'B{i}'] for i in range(0, header['N_BFIELD'])]
    
    if B_field is None:
        return 0
    else:
        try:
            B_field_idx = b_field_list.index(B_field)
        except ValueError:
            raise ValueError(f"B-field {B_field} not found in the list.")
    return B_field_idx

def confirm_whether_hdu_exist_and_if_overwrite(fits_file, hdu_id, allow_overwrite=False):
    with fits.open(fits_file) as hdul:
        if hdu_id not in hdul:
            check_hdu = False
        else:
            check_hdu = True
    if check_hdu==True and allow_overwrite==True:
        print(f"HDUs {hdu_id} found in {fits_file}.", flush=True)
        print(f"Do you want to overwrite current {hdu_id} from {fits_file} to make new plot? (y/n): ", flush=True)
        delete_hdu = input(f"Do you want to remove current {hdu_id} from {fits_file} to make new plot? (y/n): ")
        if delete_hdu.lower() == 'y':
            print(f"Removing {hdu_id} from {fits_file}.")
            delete_hdu_from_fits(fits_file, hdu_id)
            check_hdu = False
        else:
            print(f"Keeping {hdu_id} in {fits_file}.")
    
    return check_hdu

def load_stat_data_from_fits_file(fits_file, hdu_id, B_field_idx, time_range=None, zero_padding_ratio=None, allow_overwrite=False):
    if hdu_id[:5] == 'STAT_':
        hdu_id = hdu_id
    else:
        hdu_id = 'STAT_' + hdu_id

    # Check if the HDU exists in the FITS file
    check_hdu = confirm_whether_hdu_exist_and_if_overwrite(fits_file, hdu_id, allow_overwrite)

    if check_hdu==False:
        print(f"HDUs {hdu_id} not found in {fits_file}.")
        print(f"Creating {hdu_id} in {fits_file}.")
        avg_std_of_time_freq_domain(fits_file, hdu_id[5:], time_range=time_range, zero_padding_ratio=zero_padding_ratio)
    
    with fits.open(fits_file) as hdul:
        data = hdul[hdu_id].data
        header = hdul[hdu_id].header

        time_N = len(data[0][0])/header['N_BFIELD']
        freq_N = len(data[3][0])/header['N_BFIELD']

        time_idx = slice(B_field_idx*int(time_N), (B_field_idx+1)*int(time_N))
        freq_idx = slice(B_field_idx*int(freq_N), (B_field_idx+1)*int(freq_N))

        time = np.array(data[0][0][time_idx])
        E_field_avg = np.array(data[1][0][time_idx])
        E_field_std = np.array(data[2][0][time_idx])
        freq = np.array(data[3][0][freq_idx])
        fft_avg = np.array(data[4][0][freq_idx])
        fft_std = np.array(data[5][0][freq_idx])

    return time, E_field_avg, E_field_std, freq, fft_avg, fft_std

def plot_avg_std_from_fits(fits_file, hdu_id, B_field=None, time_range=None, plot_only_positive_freq=True, freq_range = None, zero_padding_ratio=None, title=None, save_fig=False, save_path=None):
    
    B_field_idx = find_B_field_idx(fits_file, hdu_id, B_field)
    
    # Extract time and frequency data
    time, E_field_avg, E_field_std, freq, fft_avg, fft_std = load_stat_data_from_fits_file(fits_file, hdu_id, B_field_idx, time_range, zero_padding_ratio, allow_overwrite=True)

    # Plot time-domain data
    plt.figure(figsize=(10, 4))
    plt.plot(time, E_field_avg, label='Average E-field')
    plt.fill_between(time, E_field_avg - E_field_std, E_field_avg + E_field_std, alpha=0.2, label='Standard Deviation')
    plt.xlabel('Time (ps)')
    plt.ylabel('E-field (arb. units)')
    if title is not None:
        plt.title(title)
    else:
        plt.title('E-field time-domain data')
    plt.legend()
    plt.grid()
    plt.tight_layout()

    if save_path is None:
        save_path = './'

    if save_fig:
        plt.savefig(save_path + f'/{title}_E_field_time_domain.png')
    else:
        plt.show()


    # Plot frequency-domain data
    plt.figure(figsize=(10, 4))
    if plot_only_positive_freq:
        freq_mask = freq > 0
        freq = freq[freq_mask]
        fft_avg = fft_avg[freq_mask]
        fft_std = fft_std[freq_mask]
    if freq_range is not None:
        freq_mask = (freq >= freq_range[0]) & (freq <= freq_range[1])
        freq = freq[freq_mask]
        fft_avg = fft_avg[freq_mask]
        fft_std = fft_std[freq_mask]
    plt.plot(freq, fft_avg, label='Average FFT')
    plt.fill_between(freq, fft_avg - fft_std, fft_avg + fft_std, alpha=0.2, label='FFT Std Dev')
    plt.xlabel('Frequency (THz)')
    plt.ylabel('Amplitude (arb. units)')
    if title is not None:
        plt.title(title)
    else:
        plt.title('FFT of E-field')
    plt.legend()
    plt.grid()
    plt.tight_layout()

    if save_path is None:
        save_path = './'

    if save_fig:
        plt.savefig(save_path + f'/{title}_FFT.png')
    else:
        plt.show()
    # Close all plots
    plt.close('all')
    print(f"Plots saved to {save_path}.")

def plot_transmission_spec(fits_file, hdu_id_tr, hdu_id_ref, B_field=None, plot_only_positive_freq=True, freq_range = None, title=None, save_fig=False, save_path=None):
    
    B_field_idx_ref = find_B_field_idx(fits_file, hdu_id_ref, B_field=0.)
    B_field_idx_tr = find_B_field_idx(fits_file, hdu_id_tr, B_field)
    
    # Extract time and frequency data
    time, E_field_avg_ref, E_field_std_ref, freq, fft_avg_ref, fft_std_ref = load_stat_data_from_fits_file(fits_file, hdu_id_ref, B_field_idx_ref, allow_overwrite=False)
    time, E_field_avg_tr, E_field_std_tr, freq, fft_avg_tr, fft_std_tr = load_stat_data_from_fits_file(fits_file, hdu_id_tr, B_field_idx_tr, allow_overwrite=False)
    # Calculate transmission spectrum
    transmission_avg = np.abs(fft_avg_tr / fft_avg_ref) 
    transmission_std = np.array(transmission_avg * ((fft_std_tr/fft_avg_tr)**2 + (fft_std_ref/fft_avg_ref)**2)**0.5)
    
    plt.figure(figsize=(10, 4))
    if plot_only_positive_freq:
        freq_mask = freq > 0
        freq = freq[freq_mask]
        transmission_avg = transmission_avg[freq_mask]
        transmission_std = transmission_std[freq_mask]
    if freq_range is not None:
        freq_mask = (freq >= freq_range[0]) & (freq <= freq_range[1])
        freq = freq[freq_mask]
        transmission_avg = transmission_avg[freq_mask]
        transmission_std = transmission_std[freq_mask]
    plt.plot(freq, transmission_avg, label='Transmission Spectrum')
    plt.fill_between(freq, transmission_avg - transmission_std, transmission_avg + transmission_std, alpha=0.2, label='Transmission Std Dev')
    plt.xlabel('Frequency (THz)')
    plt.ylabel('Transmission (arb. units)')
    if title is not None:
        plt.title(title)
    else:
        plt.title('Transmission Spectrum')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if save_path is None:
        save_path = './'
    if save_fig:
        plt.savefig(save_path + f'/{title}_Transmission_Spectrum.png')
        print(f"Plots saved to {save_path}.")
    else:
        plt.show()
    # Close all plots
    plt.close('all')
    
