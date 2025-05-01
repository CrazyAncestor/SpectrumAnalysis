import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import matplotlib.cm as cm

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

def extract_data_from_files(data_dir, output_fits_path):
    data_dict = {}
    file_pattern = os.path.join(data_dir, '*T.txt')
    files = glob.glob(file_pattern)

    for file_path in files:
        filename = os.path.basename(file_path)

        # Extract magnetic field value from filename
        match = re.search(r'_([-+]?\d*\.?\d+)T\.txt$', filename)
        if not match:
            print(f"Skipping file (no match): {filename}")
            continue

        field_value = float(match.group(1))

        try:
            with open(file_path, 'r') as f:
                header_line = f.readline().strip()
                identifier_line = f.readline().strip()  # not used, but read to skip

            # Parse metadata from header
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

            # Load data from remaining lines
            E_field = np.loadtxt(file_path, skiprows=2)

            # Generate time array in ps
            if step_fs is not None and start_ps is not None:
                time = start_ps + np.arange(len(E_field)) * step_fs * 1e-3  # fs → ps
                combined = np.column_stack((time, E_field))
            else:
                combined = np.column_stack((np.zeros_like(E_field), E_field))  # fallback if parsing failed

            # Store in structured dict
            data_dict[field_value] = {
                "step_fs": step_fs,
                "start_ps": start_ps,
                "delay_ms": delay_ms,
                "data": combined  # 2 columns: time and E_field
            }

        except Exception as e:
            print(f"Error loading {filename}: {e}")
        
    save_data_to_fits(data_dict, output_fits_path)
    return data_dict

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
                ifimport plot_time_domain:
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