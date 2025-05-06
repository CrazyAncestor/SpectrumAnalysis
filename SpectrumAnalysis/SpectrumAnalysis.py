import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import matplotlib.cm as cm
from collections import defaultdict

def save_data_to_fits(data_dict, output_path):
    hdus = [fits.PrimaryHDU()]  # Start with an empty primary HDU

    for field_value, info in sorted(data_dict.items()):
        # Prepare header
        hdr = fits.Header()
        hdr['B_FIELD'] = field_value
        if info["step_fs"] is not None:
            hdr['STEPFS'] = info["step_fs"]
        if info["start_ps"] is not None:
            hdr['STARTPS'] = info["start_ps"]
        if info["delay_ms"] is not None:
            hdr['DELAYMS'] = info["delay_ms"]

        # Create ImageHDU (1D data)
        hdu = fits.ImageHDU(data=info["data"], header=hdr, name=f'{field_value:.3f}T')
        hdus.append(hdu)
    # Write to file
    hdul = fits.HDUList(hdus)
    hdul.writeto(output_path, overwrite=True)
    print(f"Saved to {output_path}")
    
def load_data_from_fits(fits_path, print_summary=False):
    data_dict = {}

    with fits.open(fits_path) as hdul:
        for hdu in hdul[1:]:  # Skip PrimaryHDU
            header = hdu.header
            data = hdu.data

            # Extract metadata
            field_value = header.get('B_FIELD')
            step_fs = header.get('STEPFS')
            start_ps = header.get('STARTPS')
            delay_ms = header.get('DELAYMS')

            data_dict[field_value] = {
                "step_fs": step_fs,
                "start_ps": start_ps,
                "delay_ms": delay_ms,
                "data": data
            }

            # Print summary info
            if print_summary:
                print(f"{field_value} T → step: {step_fs} fs, start: {start_ps} ps, delay: {delay_ms} ms, points: {len(data)}")

    return data_dict

def extract_data_from_files(data_dir):
    grouped_data = defaultdict(dict)
    
    def extract_data_from_file(file_path, grouped_data, field_dependent, save_each_scan = False):
        filename = os.path.basename(file_path)
        
        try:
            with open(file_path, 'r') as f:
                header_line = f.readline().strip()
                identifier_line = f.readline().strip()
            
            # Extract from the header line
            header_match = re.search(
                r'step(\d+)fs_\d+pnts_from([-+]?\d+)ps_delay(\d+)ms',
                header_line
            )
            if header_match:
                step_fs = int(header_match.group(1))
                start_ps = int(header_match.group(2))
                delay_ms = int(header_match.group(3))
            else:
                print(f"Warning: couldn't parse header in {filename}")
                step_fs = start_ps = delay_ms = None

            # Determine if the file is saved for each scan or not
            if save_each_scan:
                # Extract scan number from filename
                match = re.search(r'_scan(\d+)', filename)
                if not match:
                    print(f"Skipping save each scan file (no match): {filename}")
                    return
                scan_number = int(match.group(1))

                # Extract data of E-field and time from the file
                data = np.loadtxt(file_path, skiprows=2)
                
                if  field_dependent:
                    E_field = data[:, 1]
                    # Extract field value from filename
                    match = re.search(r'_([-+]?\d*\.?\d+)T', filename)
                    if not match:
                        print(f"Skipping T file (no match): {filename}")
                        return
                    field_value = float(match.group(1))
                    if field_value not in grouped_data[identifier_line]:
                        time = data[:, 0]
                        combined = np.column_stack((time, E_field))
                        grouped_data[identifier_line][field_value] = {
                            "step_fs": step_fs,
                            "start_ps": start_ps,
                            "delay_ms": delay_ms,
                            "data": combined
                        }
                    else:
                        grouped_data[identifier_line][field_value]["data"] = np.column_stack((grouped_data[identifier_line][field_value]["data"], E_field))
                else:
                    time = data[:, 0]
                    E_field = data[:, 1]
                    combined = np.column_stack((time, E_field))
                    grouped_data[identifier_line][scan_number] = {
                        "step_fs": step_fs,
                        "start_ps": start_ps,
                        "delay_ms": delay_ms,
                        "data": combined
                    }
            else:
                # Extract data of E-field from the file
                E_field = np.loadtxt(file_path, skiprows=2)

                if step_fs is not None and start_ps is not None:
                    time = start_ps + np.arange(len(E_field)) * step_fs * 1e-3
                    combined = np.column_stack((time, E_field))
                else:
                    combined = np.column_stack((np.zeros_like(E_field), E_field))
                
                # Determine if this is a field-dependent file
                if field_dependent:
                    # Extract field value from filename
                    match = re.search(r'_([-+]?\d*\.?\d+)T\.txt$', filename)
                    if not match:
                        print(f"hello Skipping T file (no match): {filename}")
                        return
                    field_value = float(match.group(1))
                else:
                    field_value = 0.0

                grouped_data[identifier_line][field_value] = {
                    "step_fs": step_fs,
                    "start_ps": start_ps,
                    "delay_ms": delay_ms,
                    "data": combined
                }

        except Exception as e:
            print(f"Error loading T file {filename}: {e}")
    
    # --- Process field-dependent but not saved each scan files ---
    field_dependent_not_saved_each_scan_files = [
        f for f in glob.glob(os.path.join(data_dir, '_[0-9]*T.txt'))
        if any(c.isdigit() for c in os.path.basename(f)) and 'T' in os.path.basename(f)
    ]
    for file_path in field_dependent_not_saved_each_scan_files:
        extract_data_from_file(file_path, grouped_data=grouped_data, field_dependent=True, save_each_scan=False)

    # --- Process save-each-scan files ---
    all_scan_files = glob.glob(os.path.join(data_dir, '*_scan*.txt'))

    save_each_scan_files = []
    field_dependent_and_save_each_scan_files = []

    for file_path in all_scan_files:
        filename = os.path.basename(file_path)

        if re.search(r'_[\d.]+T_scan\d+', filename):
            field_dependent_and_save_each_scan_files.append(file_path)
        else:
            save_each_scan_files.append(file_path)


    # --- Process: non-field-dependent scan files ---
    for file_path in save_each_scan_files:
        extract_data_from_file(file_path, grouped_data=grouped_data, field_dependent=False, save_each_scan=True)

    # --- Process: field-dependent scan files ---
    for file_path in field_dependent_and_save_each_scan_files:
        extract_data_from_file(file_path, grouped_data=grouped_data, field_dependent=True, save_each_scan=True)


    # --- Process *_Aperture.txt files ---
    aperture_files = glob.glob(os.path.join(data_dir, '*_Aperture.txt'))
    for file_path in aperture_files:
        extract_data_from_file(file_path, grouped_data=grouped_data, field_dependent=False)

    # --- Save grouped data ---
    for identifier, full_data in grouped_data.items():
        output_fits_path = identifier + '.fits'
        save_data_to_fits(full_data, output_fits_path)

    return grouped_data

def plot_Efields_from_fits(fits_path, title="E-field vs Time"):
    # Open the FITS file
    with fits.open(fits_path) as hdul:
        plt.figure(figsize=(10, 6))

        # Iterate over the HDUs (skip the PrimaryHDU)
        for hdu in hdul[1:]:
            header = hdu.header
            data = hdu.data

            # Extract field_value from header
            field_value = header.get('B_FIELD')

            # Assuming the data is 2D: time in first column, E_field in second column
            time = data[:, 0]  # ps
            if data.shape[1] > 2:
                E_field = np.average(data[:, 1:], axis=1)  # average over multiple columns
            else:
                E_field = data[:, 1]  # arbitrary units

            # Plot the data
            plt.plot(time, E_field, label=f'{field_value:.3f} T')

        # Plot formatting
        plt.xlabel("Time (ps)")
        plt.ylabel("E-field (arb. units)")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def fft_with_zero_padding(time, td_data, zero_padding_ratio = None):
    if zero_padding_ratio is None:
        N = len(td_data)
        dt = time[1] - time[0]  # Time step (assumed constant)
        freq = np.fft.fftfreq(N, dt)  # Frequency values
        td_data_fft = np.fft.fft(td_data)  # FFT of td_data
        td_data_fft_mag = np.abs(td_data_fft)  # Magnitude of FFT
    else:
        N = int(len(td_data) * zero_padding_ratio)
        add_mum = N - len(td_data)
        dt = time[1] - time[0]
        td_data_new = np.concatenate([np.array(td_data), np.zeros(add_mum)])
        freq = np.fft.fftfreq(N, dt)  # Frequency values
        td_data_fft = np.fft.fft(td_data_new)  # FFT of td_data
        td_data_fft_mag = np.abs(td_data_fft)  # Magnitude of FFT

    return N, freq, td_data_fft_mag

def plot_fft_from_fits(fits_path, zero_padding_ratio = 1., time_range=None, positive_B_field_or_not = True, title="FFT of E-field", plot_time_domain=False):
    # Open the FITS file
    with fits.open(fits_path) as hdul:
        plt.figure(figsize=(10, 6))

        # Get the colormap
        colormap = cm.viridis  # You can change this to 'plasma', 'inferno', etc.
        num_colors = len(hdul) - 1  # Number of datasets (skip PrimaryHDU)

        # Iterate over the HDUs (skip the PrimaryHDU)
        for idx, hdu in enumerate(hdul[1:]):
            header = hdu.header
            data = hdu.data

            # Extract field_value from header
            field_value = header.get('B_FIELD')

            if positive_B_field_or_not and field_value < 0:
                continue
            elif not positive_B_field_or_not and field_value > 0:
                continue                
            else:
                # Assuming the data is 2D: time in first column, E_field in second column
                time = data[:, 0]  # ps
                if data.shape[1] > 2:
                    E_field = np.average(data[:, 1:], axis=1)  # average over multiple columns
                else:
                    E_field = data[:, 1]  # arbitrary units

                # Sort by time
                sorted_indices = np.argsort(time)
                time = time[sorted_indices]
                E_field = E_field[sorted_indices]

                if time_range is not None:
                    # Filter time and E_field based on the specified range
                    mask = (time >= time_range[0]) & (time <= time_range[1])
                    time = time[mask]
                    E_field = E_field[mask]

                # Perform FFT
                N, freq, E_field_fft_mag = fft_with_zero_padding(time, E_field, zero_padding_ratio=zero_padding_ratio)

                # Only keep the positive frequencies and their magnitudes
                positive_freq = freq[:N // 2]
                positive_fft_mag = E_field_fft_mag[:N // 2]

                # Select the color for this field_value using colormap
                color = colormap(idx / num_colors)  # Normalize index to the colormap range

                # Plot time domain data (optional)
                if plot_time_domain:
                    plt.subplot(2, 1, 1)  # Time domain plot (top)
                    plt.plot(time, E_field, label=f'{field_value:.3f} T', color=color)
                    plt.xlabel("Time (ps)")
                    plt.ylabel("E-field (arb. units)")
                    plt.title("Time Domain")
                    plt.legend()
                    plt.grid(True)

                # Plot FFT (frequency domain) for positive frequencies only
                plt.subplot(2, 1, 2)  # Frequency domain plot (bottom)
                plt.plot(positive_freq, positive_fft_mag, label=f'{field_value:.3f} T', color=color)
                plt.xlabel("Frequency (THz)")
                plt.ylabel("Magnitude")
                plt.title(title)
                plt.legend()
                plt.grid(True)

        plt.tight_layout()
        plt.show()

def plot_transmission_spectrum(reference_fits, sample_fits, freq_range = None, zero_padding_ratio = 1., positive_B_field_or_not = True, title="Transmission Spectrum"):
    # Open both FITS files
    with fits.open(reference_fits) as ref_hdul, fits.open(sample_fits) as samp_hdul:
        plt.figure(figsize=(10, 6))
        colormap = cm.plasma
        num_colors = len(samp_hdul) - 1  # Skip PrimaryHDU

        for idx, samp_hdu in enumerate(samp_hdul[1:]):
            ref_data = ref_hdul[1].data
            samp_data = samp_hdu.data
            samp_B = samp_hdu.header.get("B_FIELD")
            if positive_B_field_or_not and samp_B < 0:
                continue
            elif not positive_B_field_or_not and samp_B > 0:
                continue   

            # Extract time and E-field
            time = ref_data[:, 0]
            ref_E = ref_data[:, 1]
            samp_E = samp_data[:, 1]

            N, freq, ref_mag = fft_with_zero_padding(time, ref_E, zero_padding_ratio=zero_padding_ratio)
            N, freq, samp_mag = fft_with_zero_padding(time, samp_E, zero_padding_ratio=zero_padding_ratio)
            trans = samp_mag/ref_mag  # Avoid division by zero

            color = colormap(idx / num_colors)
            label = f"{samp_B:.3f} T"

            plt.plot(freq[:N // 2], trans[:N // 2], label=label, color=color)

        plt.xlabel("Frequency (THz)")
        plt.ylabel("Transmission (arb. units)")
        if freq_range is not None:
            plt.xlim(freq_range)
        plt.ylim(0, 1)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def add_data_value(fits1, fits2, output_path):
    dataset1 = load_data_from_fits(fits1)
    dataset2 = load_data_from_fits(fits2)
    data_added = defaultdict(dict)

    for field_value, entry in dataset1.items():
        combined = np.column_stack((dataset1[field_value]['data'][:,0], dataset1[field_value]['data'][:,1] + dataset2[field_value]['data'][:,1]))
        data_added[field_value] = {
                    "step_fs": dataset1[field_value]['step_fs'],
                    "start_ps": dataset1[field_value]['start_ps'],
                    "delay_ms": dataset1[field_value]['delay_ms'],
                    "data": combined
                }
    save_data_to_fits(data_added, output_path)
    return data_added

def subtract_data_value(fits1, fits2, output_path):
    dataset1 = load_data_from_fits(fits1)
    dataset2 = load_data_from_fits(fits2)
    data_subtracted = defaultdict(dict)

    for field_value, entry in dataset1.items():
        combined = np.column_stack((dataset1[field_value]['data'][:,0], dataset1[field_value]['data'][:,1] - dataset2[field_value]['data'][:,1]))
        data_subtracted[field_value] = {
                    "step_fs": dataset1[field_value]['step_fs'],
                    "start_ps": dataset1[field_value]['start_ps'],
                    "delay_ms": dataset1[field_value]['delay_ms'],
                    "data": combined
                }
    save_data_to_fits(data_subtracted, output_path)
    return data_subtracted

def rotate_angle_vs_major_axis(Ex_fits, Ey_fits, angle_in_deg, output_path_major, output_path_minor):
    Ex_dataset = load_data_from_fits(Ex_fits)
    Ey_dataset = load_data_from_fits(Ey_fits)

    data_major = defaultdict(dict)
    data_minor = defaultdict(dict)

    angle_in_rad = np.deg2rad(angle_in_deg)
    for field_value, entry in Ex_dataset.items():
        combined_major = np.column_stack((Ex_dataset[field_value]['data'][:,0], Ex_dataset[field_value]['data'][:,1] *np.sin(angle_in_rad) - Ey_dataset[field_value]['data'][:,1]*np.cos(angle_in_rad)))
        combined_minor = np.column_stack((Ex_dataset[field_value]['data'][:,0], Ex_dataset[field_value]['data'][:,1] *np.cos(angle_in_rad) + Ey_dataset[field_value]['data'][:,1]*np.sin(angle_in_rad)))
        data_major[field_value] = {
                    "step_fs": Ex_dataset[field_value]['step_fs'],
                    "start_ps": Ex_dataset[field_value]['start_ps'],
                    "delay_ms": Ex_dataset[field_value]['delay_ms'],
                    "data": combined_major
                }
        data_minor[field_value] = {
                    "step_fs": Ey_dataset[field_value]['step_fs'],
                    "start_ps": Ey_dataset[field_value]['start_ps'],
                    "delay_ms": Ey_dataset[field_value]['delay_ms'],
                    "data": combined_minor
                }

    save_data_to_fits(data_major, output_path_major)
    save_data_to_fits(data_minor, output_path_minor)
    return data_major, data_minor

def plot_avg_std_from_fits(fits_file, time_range=None, B_field=None):
    metadata = load_data_from_fits(fits_file)
    
    # For non-field-dependent data, we assume each hdu corresponds to a different scan
    if metadata[1]['data'].shape[1]==2:
        time = metadata[1]['data'][:, 0]
        E_field = [metadata[key]['data'][:, 1] for key in metadata]

        if time_range is not None:
        # Filter time and E_field based on the specified range
            mask = (time >= time_range[0]) & (time <= time_range[1])
            time = time[mask]
            E_field = E_field[mask]
        
        # Time-domain average and std
        E_field_stack = np.stack(E_field)
        E_field_avg = np.mean(E_field_stack, axis=0)
        E_field_std = np.std(E_field_stack, axis=0)
    else:
        # For field-dependent data, we assume each hdu corresponds to a different field value
        time = metadata[B_field]['data'][:, 0]
        E_field = metadata[B_field]['data'][:, 1:]
        if time_range is not None:
        # Filter time and E_field based on the specified range
            mask = (time >= time_range[0]) & (time <= time_range[1])
            time = time[mask]
            e = []
            for i in range(E_field.shape[1]):
                e.append(E_field[:,i][mask])
            E_field = np.array(e).T

        # Time-domain average and std
        E_field_avg = np.mean(E_field, axis=1)
        E_field_std = np.std(E_field, axis=1)

    # Plot time-domain data
    plt.figure(figsize=(10, 4))
    plt.plot(time, E_field_avg, label='Average E-field')
    plt.fill_between(time, E_field_avg - E_field_std, E_field_avg + E_field_std, alpha=0.2, label='Standard Deviation')
    plt.xlabel('Time (ps)')
    plt.ylabel('E-field (arb. units)')
    plt.title('E-field time-domain data')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # FFT computation
    if metadata[1]['data'].shape[1]==2:
        fft_all = [np.fft.fft(e) for e in E_field]
    else:
        fft_all = [np.fft.fft(E_field[:,i]) for i in range(E_field.shape[1])]
    fft_stack = np.stack(fft_all)
    fft_avg = np.abs(np.mean(fft_stack, axis=0))
    fft_std = np.abs(np.std(fft_stack, axis=0))
    

    freqs = np.fft.fftfreq(len(time), d=(time[1] - time[0]))  # Frequency axis

    # sort frequencies and FFT values
    sorted_indices = np.argsort(freqs)
    freqs = freqs[sorted_indices]
    fft_avg = fft_avg[sorted_indices]  
    fft_std = fft_std[sorted_indices]

    # Plot frequency-domain data
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, fft_avg, label='Average FFT')
    plt.fill_between(freqs, fft_avg - fft_std, fft_avg + fft_std, alpha=0.2, label='FFT Std Dev')
    plt.xlabel('Frequency (THz)')
    plt.ylabel('Amplitude (arb. units)')
    plt.title('FFT of E-field')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    return E_field_avg, E_field_std, fft_avg, fft_std, freqs