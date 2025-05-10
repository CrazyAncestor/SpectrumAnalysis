import numpy as np
from astropy.io import fits

def avg_std_of_time_freq_domain(fits_file, hdu_id = None, time_range=None, B_field=None, zero_padding_ratio=None):

    def avg_std(data, time_range):
        time = data[ :, 0]
        E_field = data[ :, 1:]

        # Apply time range filtering
        if time_range is not None:
            t_min, t_max = time_range
            time_mask = (time >= t_min) & (time <= t_max)

            # Apply mask along time axis
            # This assumes time is the same across rows (if not, mask must be applied individually)
            time = time[time_mask]
            E_field = E_field[time_mask, :]

        # Time-domain average and std
        E_field_avg = np.mean(E_field, axis=1)
        E_field_std = np.std(E_field, axis=1)

        N, freq, fft_avg, fft_std =  fft_with_zero_padding(time, E_field, zero_padding_ratio = zero_padding_ratio)

        return np.array(time), np.array(E_field_avg), np.array(E_field_std), np.array(freq), np.array(fft_avg), np.array(fft_std)
    
    with fits.open(fits_file, mode='update') as hdul:
        hdu = hdul[hdu_id]
        times = []
        E_field_avgs = []
        E_field_stds = []
        freqs = []
        fft_avgs = []
        fft_stds = []
        for B_field_idx in range(hdu.data.shape[0]):
            data = hdu.data[B_field_idx, :, :]
            time, E_field_avg, E_field_std, freq, fft_avg, fft_std = avg_std(data, time_range)
            times = np.append(times, time)
            E_field_avgs = np.append(E_field_avgs, E_field_avg)
            E_field_stds = np.append(E_field_stds, E_field_std)

            freqs = np.append(freqs, freq)
            fft_avgs = np.append(fft_avgs, fft_avg)
            fft_stds = np.append(fft_stds, fft_std)
        
        print(f"Times.shape: {times.shape}")
        
        # Flatten the arrays
        stat_data = [times, E_field_avgs, E_field_stds, freqs, fft_avgs, fft_stds]
        stat_data = [arr.flatten().astype(np.float32) for arr in stat_data]

        # Convert to a format that fits understands: an object array
        col = fits.Column(name='STAT_'+hdu_id, format='PE()', array=np.array(stat_data, dtype=object))

        # Make a binary table
        hdu_new = fits.BinTableHDU.from_columns([col])
        hdu_new.header = hdu.header.copy()
        hdu_new.header['EXTNAME'] = 'STAT_' + str(hdu_id)
        hdu_new.header['HDU_TYPE'] = 'STAT_' + str(hdu_id)
        print(f"Adding {hdu_new.name} to the FITS file.")
        print(hdu_new.header)

        # Append the new HDU to the original HDU list
        hdul.append(hdu_new.copy())


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
    fft_avg = np.abs(np.mean(fft_result, axis=1))                 # Shape: (scan,)
    fft_std = np.abs(np.std(fft_result, axis=1))                # Shape: (scan,)

    # sort the frequency values and corresponding FFT results
    sort_indices = np.argsort(freq)
    freq = freq[sort_indices]
    fft_avg = fft_avg[sort_indices]
    fft_std = fft_std[sort_indices]


    return N, freq, fft_avg, fft_std