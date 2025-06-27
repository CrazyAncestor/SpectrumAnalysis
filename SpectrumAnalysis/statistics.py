import numpy as np
from astropy.io import fits
from .util import read_image_hdu, write_data_to_bin_hdu, confirm_whether_hdu_exist_and_if_overwrite

def statistical_analysis(fits_file, hdu_raw_data_name, plot_time_range=None, fft_time_range=None, B_field=None, zero_padding_ratio=None):
    raw_data_header, raw_data = read_image_hdu(fits_file, hdu_raw_data_name)

    times = []
    E_field_avgs = []
    E_field_stds = []
    freqs = []
    fft_avg_reals = []
    fft_avg_imags = []
    fft_stds = []

    for B_field_namex in range(raw_data.shape[0]):
        time, E_field_avg, E_field_std, freq, fft_avg, fft_std \
        = avg_std_of_time_freq_data(raw_data = raw_data[B_field_namex,:,:], reference_time=raw_data[0,:,0], plot_time_range=plot_time_range, fft_time_range=fft_time_range, zero_padding_ratio=zero_padding_ratio)
        times = np.append(times, time)
        E_field_avgs = np.append(E_field_avgs, E_field_avg)
        E_field_stds = np.append(E_field_stds, E_field_std)

        freqs = np.append(freqs, freq)
        fft_avg_reals = np.append(fft_avg_reals, np.real(fft_avg))
        fft_avg_imags = np.append(fft_avg_imags, np.imag(fft_avg))
        fft_stds = np.append(fft_stds, fft_std)

    stat_data = [times, E_field_avgs, E_field_stds, freqs, fft_avg_reals, fft_avg_imags, fft_stds]

    hdu_name_stat = f'STAT_{hdu_raw_data_name}'
    hdu_exist, overwrite_or_not = confirm_whether_hdu_exist_and_if_overwrite(fits_file, hdu_name_stat, allow_overwrite=True)
    if hdu_exist==False or overwrite_or_not==True:
        write_data_to_bin_hdu(fits_file = fits_file, hdu_new_name = hdu_name_stat, hdu_new_type='STAT', hdu_new_header=raw_data_header, stat_data=stat_data)

def avg_std_of_time_freq_data(raw_data, reference_time, plot_time_range, fft_time_range, zero_padding_ratio):
        time = reference_time # Ensure the time is homogeneous
        E_field = raw_data[ :, 1:]

        
        # Apply time range filtering for FFT transform
        if fft_time_range is not None:

            t_min, t_max = fft_time_range

            # Check if fft_time_range is within the range of time
            if t_min < time[0] or t_max > time[-1]:
                raise ValueError(f"fft_time_range {fft_time_range} is out of bounds of the time data.")
            
            # Apply time range filtering
            time_mask = (time >= t_min) & (time <= t_max)
            time_fft = time[time_mask]
            E_field_fft = E_field[time_mask, :]
        else:
            time_fft = time
            E_field_fft = E_field

        N, freq, fft_avg, fft_std =  fft_with_zero_padding(time_fft, E_field_fft, zero_padding_ratio = zero_padding_ratio)
        
        
        if plot_time_range is not None:

            t_min, t_max = plot_time_range

            # Check if plot_time_range is within the range of time
            if t_min < time[0] or t_max > time[-1]:
                raise ValueError(f"plot_time_range {plot_time_range} is out of bounds of the time data.")
            
            # Apply time range filtering for time-domain average and std
            time_mask = (time >= t_min) & (time <= t_max)
            time = time[time_mask]
            E_field = E_field[time_mask, :]

        # Time-domain average and std
        E_field_avg = np.mean(E_field, axis=1)
        E_field_std = np.std(E_field, axis=1)

        return np.array(time), np.array(E_field_avg), np.array(E_field_std), np.array(freq), np.array(fft_avg), np.array(fft_std)

def fft_with_zero_padding(time, td_data, zero_padding_ratio = None):
    if zero_padding_ratio is None:
        N = len(time)
        td_data_new = np.copy(td_data)
        
    else:
        N = int(len(time) * zero_padding_ratio)
        add_mum = N - len(time)
        td_data_new = np.concatenate([np.array(np.copy(td_data)), np.zeros((add_mum, td_data.shape[1]))], axis=0)
    
    dt = time[1] - time[0]  # Time step (assumed constant)
    freq = np.fft.fftfreq(N, dt)  # Frequency values
    # Frequency-domain (FFT)
    fft_result = np.fft.fft(td_data_new, axis=0)                 # Shape: (frequencies, scan)

    # Average and std across frequency bins per component
    fft_avg = np.mean(fft_result, axis=1)                 # Shape: (scan,)
    fft_std = np.std(fft_result, axis=1)                # Shape: (scan,)

    # sort the frequency values and corresponding FFT results
    sort_indices = np.argsort(freq)
    freq = freq[sort_indices]
    fft_avg = fft_avg[sort_indices]
    fft_std = fft_std[sort_indices]

    return N, freq, fft_avg, fft_std