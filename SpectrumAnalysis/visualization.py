import matplotlib.pyplot as plt
import numpy as np
from .util import raw_stat_hdu_name, read_stat_hdu
from .statistics import statistical_analysis

def plot_statistics(fits_file, hdu_id, B_field=0., calculate_from_raw_data=True, time_range=None, plot_only_positive_freq=True, freq_range = None, zero_padding_ratio=None, title=None, save_fig=False, save_path=None):
    
    # First do statistical analysis
    hdu_raw_data_id, hdu_stat_id = raw_stat_hdu_name(hdu_id)
    if calculate_from_raw_data:
        statistical_analysis(fits_file, hdu_raw_data_id, time_range, B_field, zero_padding_ratio)
    # Then read the hdu data
    times, E_field_avgs, E_field_stds, freqs, fft_avg_reals, fft_avg_imags, fft_stds, B_field_values, _ = read_stat_hdu(fits_file, hdu_stat_id, B_field)


    # Plot time-domain data
    plt.figure(figsize=(10, 4))

    plt.figure(figsize=(10, 4))
    for i in range(freqs.shape[0]):
        plt.plot(times[i], E_field_avgs[i], label=f'B={B_field_values[i]}T')
        plt.fill_between(times[i], E_field_avgs[i] - E_field_stds[i], E_field_avgs[i] + E_field_stds[i], alpha=0.2)
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

    for i in range(freqs.shape[0]):
        freq = freqs[i]
        fft_avg_real = fft_avg_reals[i]
        fft_avg_imag = fft_avg_imags[i]
        fft_std = fft_stds[i]
        if plot_only_positive_freq:
            freq_mask = freq > 0
            freq = freq[freq_mask]
            fft_avg_real = fft_avg_real[freq_mask]
            fft_avg_imag = fft_avg_imag[freq_mask]
            fft_std = fft_std[freq_mask]
        if freq_range is not None:
            freq_mask = (freq >= freq_range[0]) & (freq <= freq_range[1])
            freq = freq[freq_mask]
            fft_avg_real = fft_avg_real[freq_mask]
            fft_avg_imag = fft_avg_imag[freq_mask]
            fft_std = fft_std[freq_mask]
        fft_avg = np.abs(fft_avg_real**2 +fft_avg_imag**2)**0.5
        fft_std = np.abs(fft_std)
        plt.plot(freq, fft_avg, label=f'B={B_field_values[i]}T')
        plt.fill_between(freq, fft_avg - fft_std, fft_avg + fft_std, alpha=0.2)
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

def plot_SN_ratio(fits_file, hdu_id, B_field=0., plot_only_positive_freq=True, freq_range = None, title=None, save_fig=False, save_path=None):
    
    # Extract HDU data
    hdu_raw_data_id, hdu_stat_id = raw_stat_hdu_name(hdu_id)
    times, E_field_avgs, E_field_stds, freqs, fft_avg_reals, fft_avg_imags, fft_stds, B_field_values, _ = read_stat_hdu(fits_file, hdu_stat_id, B_field)


    # Plot time-domain SN ratio
    plt.figure(figsize=(10, 4))

    plt.figure(figsize=(10, 4))
    for i in range(freqs.shape[0]):
        plt.plot(times[i], E_field_avgs[i]/E_field_stds[i], label=f'B={B_field_values[i]}T')
    plt.xlabel('Time (ps)')
    plt.ylabel('E-field S/N ratio (arb. units)')
    if title is not None:
        plt.title(title)
    else:
        plt.title('E-field time-domain data S/N ratio')
    plt.legend()
    plt.grid()
    plt.tight_layout()

    if save_path is None:
        save_path = './'

    if save_fig:
        plt.savefig(save_path + f'/{title}_E_field_time_domain_sn.png')
    else:
        plt.show()


    # Plot frequency-domain data
    plt.figure(figsize=(10, 4))

    for i in range(freqs.shape[0]):
        freq = freqs[i]
        fft_avg_real = fft_avg_reals[i]
        fft_avg_imag = fft_avg_imags[i]
        fft_std = fft_stds[i]
        if plot_only_positive_freq:
            freq_mask = freq > 0
            freq = freq[freq_mask]
            fft_avg_real = fft_avg_real[freq_mask]
            fft_avg_imag = fft_avg_imag[freq_mask]
            fft_std = fft_std[freq_mask]
        if freq_range is not None:
            freq_mask = (freq >= freq_range[0]) & (freq <= freq_range[1])
            freq = freq[freq_mask]
            fft_avg_real = fft_avg_real[freq_mask]
            fft_avg_imag = fft_avg_imag[freq_mask]
            fft_std = fft_std[freq_mask]
        fft_avg = np.abs(fft_avg_real**2 +fft_avg_imag**2)**0.5
        fft_std = np.abs(fft_std)
        plt.plot(freq, fft_avg/fft_std, label=f'B={B_field_values[i]}T')
    plt.xlabel('Frequency (THz)')
    plt.ylabel('Amplitude S/N ratio (arb. units)')
    if title is not None:
        plt.title(title)
    else:
        plt.title('FFT S/N ratio of E-field')
    plt.legend()
    plt.grid()
    plt.tight_layout()

    if save_path is None:
        save_path = './'

    if save_fig:
        plt.savefig(save_path + f'/{title}_FFT_sn.png')
    else:
        plt.show()
    # Close all plots
    plt.close('all')
    print(f"Plots saved to {save_path}.")


def plot_transmission_spec(fits_file, hdu_tr_id, hdu_ref_id, B_field=None, plot_only_positive_freq=True, freq_range = None, title=None, save_fig=False, save_path=None):
    # Power Transmission
    # Extract time and frequency data
    times, E_field_avg_refs, E_field_std_refs, freqs, fft_avg_real_refs, fft_avg_imag_refs, fft_std_refs, B_field_values, _ = read_stat_hdu(fits_file, hdu_ref_id, B_field= 0.0)
    times, E_field_avg_trs, E_field_std_trs, freqs, fft_avg_real_trs, fft_avg_imag_trs, fft_std_trs, B_field_values, _ = read_stat_hdu(fits_file, hdu_tr_id, B_field=B_field)

    plt.figure(figsize=(10, 4))
    for i in range(freqs.shape[0]):
    # Calculate transmission spectrum
        transmission_avg = (fft_avg_real_trs[i]**2 + fft_avg_imag_trs[i]**2) / (fft_avg_real_refs[0]**2 + fft_avg_imag_refs[0]**2)
        transmission_std = 2 * np.array(transmission_avg * (np.abs(fft_std_trs[i])**2/(fft_avg_real_trs[i]**2 + fft_avg_imag_trs[i]**2) + \
                                                        np.abs(fft_std_refs[0])**2/(fft_avg_real_refs[0]**2 + fft_avg_imag_refs[0]**2) )**0.5)
        
        if plot_only_positive_freq:
            freq_mask = freqs[i] > 0
            freq = freqs[i][freq_mask]
            transmission_avg = transmission_avg[freq_mask]
            transmission_std = transmission_std[freq_mask]
        if freq_range is not None:
            freq_mask = (freq >= freq_range[0]) & (freq <= freq_range[1])
            freq = freq[freq_mask]
            transmission_avg = transmission_avg[freq_mask]
            transmission_std = transmission_std[freq_mask]
        plt.plot(freq, transmission_avg, label=f'B={B_field_values[i]}T')
        plt.fill_between(freq, transmission_avg - transmission_std, transmission_avg + transmission_std, alpha=0.2)
    
    plt.xlabel('Frequency (THz)')
    plt.ylabel('Transmission (arb. units)')
    if title is not None:
        plt.title(title)
    else:
        plt.title('Transmission Spectrum')
    plt.legend(loc = 'best')
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
    