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
    
def load_data_from_fits(fits_path):
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
            print(f"{field_value} T → step: {step_fs} fs, start: {start_ps} ps, delay: {delay_ms} ms, points: {len(data)}")

    return data_dict

def extract_data_from_files(data_dir):
    grouped_data = defaultdict(dict)

    def extract_data_from_file(file_path, grouped_data, sample_or_not):
        filename = os.path.basename(file_path)
        if sample_or_not:
            match = re.search(r'_([-+]?\d*\.?\d+)T\.txt$', filename)
            if not match:
                print(f"Skipping T file (no match): {filename}")
                return

            field_value = float(match.group(1))
        else:
            field_value = 0.0

        try:
            with open(file_path, 'r') as f:
                header_line = f.readline().strip()
                identifier_line = f.readline().strip()

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

            E_field = np.loadtxt(file_path, skiprows=2)

            if step_fs is not None and start_ps is not None:
                time = start_ps + np.arange(len(E_field)) * step_fs * 1e-3
                combined = np.column_stack((time, E_field))
            else:
                combined = np.column_stack((np.zeros_like(E_field), E_field))

            grouped_data[identifier_line][field_value] = {
                "step_fs": step_fs,
                "start_ps": start_ps,
                "delay_ms": delay_ms,
                "data": combined
            }

        except Exception as e:
            print(f"Error loading T file {filename}: {e}")
    
    # --- Process *T.txt files ---
    sample_files = glob.glob(os.path.join(data_dir, '*T.txt'))
    for file_path in sample_files:
        extract_data_from_file(file_path, grouped_data=grouped_data, sample_or_not=True)

    # --- Process *_Aperture.txt files ---
    aperture_files = glob.glob(os.path.join(data_dir, '*_Aperture.txt'))
    for file_path in aperture_files:
        extract_data_from_file(file_path, grouped_data=grouped_data, sample_or_not=False)

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

def plot_fft_from_fits(fits_path, positive_B_field_or_not = True, title="FFT of E-field", plot_time_domain=False):
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
                E_field = data[:, 1]  # arbitrary units

                # Perform FFT
                N = len(E_field)
                dt = time[1] - time[0]  # Time step (assumed constant)
                freq = np.fft.fftfreq(N, dt)  # Frequency values
                E_field_fft = np.fft.fft(E_field)  # FFT of E_field
                E_field_fft_mag = np.abs(E_field_fft)  # Magnitude of FFT

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
                plt.title("FFT Spectrum (Positive Frequencies)")
                plt.legend()
                plt.grid(True)

        plt.tight_layout()
        plt.show()

def plot_transmission_spectrum(reference_fits, sample_fits, freq_range = None, positive_B_field_or_not = True, title="Transmission Spectrum"):
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

            N = len(ref_E)
            dt = time[1] - time[0]
            freq = np.fft.fftfreq(N, dt)

            ref_fft = np.fft.fft(ref_E)
            samp_fft = np.fft.fft(samp_E)

            ref_mag = np.abs(ref_fft[:N // 2])
            samp_mag = np.abs(samp_fft[:N // 2])
            trans = samp_mag/ref_mag  # Avoid division by zero

            color = colormap(idx / num_colors)
            label = f"{samp_B:.3f} T"

            plt.plot(freq[:N // 2], trans, label=label, color=color)

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