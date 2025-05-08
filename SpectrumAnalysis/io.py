import os
import glob
import re
import numpy as np
from astropy.io import fits
from collections import defaultdict
import zipfile
import os

def extract_basic_info_and_power(raw_data_dir):
    # Step 2: Extract basic_info.txt
    basic_info_path = os.path.join(raw_data_dir, 'basic_info.txt')
    if os.path.exists(basic_info_path):
        with open(basic_info_path, 'r') as f:
            date = f.readline().strip()
            author = f.readline().strip()
    else:
        date = 'UNKNOWN'
        author = 'UNKNOWN'

    # Step 3: Extract power_log.txt
    power_log_path = os.path.join(raw_data_dir, 'power_log.txt')
    if os.path.exists(power_log_path):
        with open(power_log_path, 'r') as f:
            power_str = f.readline().strip()
            try:
                power = float(power_str)
            except ValueError:
                power = -1.0  # fallback
    else:
        power = -1.0
    return date, author, power

def process_data_zip(zip_path, metadata_filename ='metadata.fits', raw_data_dir=None, preprocessed_data_dir=None):
    # Step 1: Unzip
    unzip_file(zip_path, extract_to=raw_data_dir)

    # Step 4: Create a FITS file with this metadata in the primary header
    hdu = fits.PrimaryHDU()
    hdul = fits.HDUList([hdu])

    # Step 5: Save the FITS file
    os.makedirs(preprocessed_data_dir, exist_ok=True)
    fits_path = os.path.join(preprocessed_data_dir, metadata_filename)
    hdul.writeto(fits_path, overwrite=True)

    # Step 6: Continue processing other data
    preprocessed_filenames = extract_data_from_files(raw_data_dir, preprocessed_data_dir=preprocessed_data_dir)

    # Step 7: Print the preprocessed filenames
    print("Preprocessed files:")
    for filename in preprocessed_filenames:
        print(f" - {filename}")
    
    # Step 8: Add HDU to the FITS file
    hdu_index = 1

    while True:
        for hdu_type in ["REF", "SAMPLE"]:
            hdu_name = f"{hdu_type}{hdu_index}"
            result = add_hdu_with_prompt_message(fits_path, preprocessed_filenames, hdu_name)
            if result == "stop":
                break
        else:
            hdu_index += 1
            continue
        break


def unzip_file(zip_path, extract_to=None):
    """
    Extracts all files from a zip archive directly into the target folder,
    ignoring internal folder structure.
    
    Args:
        zip_path (str): Path to the zip file.
        extract_to (str, optional): Output folder. Defaults to zip file name (no extension).
    
    Returns:
        str: Path to the extraction directory.
    """
    if not extract_to:
        extract_to = os.path.splitext(zip_path)[0]

    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in zip_ref.infolist():
            if not member.is_dir():
                # Get just the filename without any internal folders
                filename = os.path.basename(member.filename)
                if filename:  # skip if it's an empty name (e.g., directory)
                    target_path = os.path.join(extract_to, filename)
                    with zip_ref.open(member) as source, open(target_path, 'wb') as target:
                        target.write(source.read())

    return extract_to

def add_hdu_with_prompt_message(fits_path, preprocessed_filenames, hdu_name="REF1"):
    """
    Prompts the user to select a FITS file, stacks its 2D HDUs into a 3D array,
    and appends it to the specified FITS file as a new ImageHDU.

    Parameters:
        fits_path (str): Path to the FITS file to update.
        preprocessed_filenames (list of str): List of full paths to candidate FITS files.
        hdu_name (str): Name to assign to the new HDU (e.g., 'REF1', 'SAMPLE1').
    """
    print(f"Please select the FITS file to add as HDU (press 's' to stop adding hdu) '{hdu_name}':", flush=True)

    for i, full_path in enumerate(preprocessed_filenames):
        print(f"{i}: {full_path}")

    # Validate user input
    while True:
        input_message = input("Enter the index of the file: ")
        if input_message.lower() == 's':
            print("Stopping the addition of HDU.")
            return "stop"
        try:
            index = int(input_message)
            if 0 <= index < len(preprocessed_filenames):
                selected_filename = preprocessed_filenames[int(input_message)]
                print(f"You selected: {selected_filename}")
                break
            else:
                print("Index out of range. Try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")


    # Open and process the selected FITS file
    with fits.open(selected_filename) as hdul_input:
        # Create and append the new ImageHDU
        new_hdu = fits.ImageHDU(header=hdul_input[1].header ,data=hdul_input[1].data, name=hdu_name.upper())

    with fits.open(fits_path, mode='update') as hdul:
        hdul.append(new_hdu.copy())
        hdul.flush()

    print(f"HDU '{hdu_name.upper()}' added successfully.")
    
def load_data_from_fits(fits_path, print_summary=False):
    data_dict = {}

    with fits.open(fits_path) as hdul:
        for hdu in hdul[1:]:  # Skip PrimaryHDU
            header = hdu.header
            data = hdu.data

            # Extract metadata
            B_field_value = header.get('B_FIELD')
            step_fs = header.get('STEPFS')
            start_ps = header.get('STARTPS')
            delay_ms = header.get('DELAYMS')

            data_dict[B_field_value] = {
                "step_fs": step_fs,
                "start_ps": start_ps,
                "delay_ms": delay_ms,
                "data": data
            }

            # Print summary info
            if print_summary:
                print(f"{B_field_value} T → step: {step_fs} fs, start: {start_ps} ps, delay: {delay_ms} ms, points: {len(data)}")

    return data_dict

def extract_data_from_files(raw_data_dir, preprocessed_data_dir=None):

    grouped_data = defaultdict(dict)

    def extract_data_from_file(file_path, grouped_data, B_field_dependent, save_each_scan = False):
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
                # Extract data of E-field and time from the file
                data = np.loadtxt(file_path, skiprows=2)
                E_field = data[:, 1]

                if  B_field_dependent:
                    # Extract field value from filename
                    match = re.search(r'_([-+]?\d*\.?\d+)T', filename)
                    if not match:
                        print(f"Skipping T file (no match): {filename}")
                        return
                    B_field_value = float(match.group(1))
                else:
                    B_field_value = 0.0

                # Check if the B_field_value already exists in the grouped_data
                # Then store the data in the dictionary
                if B_field_value not in grouped_data[identifier_line]:
                    time = data[:, 0]
                    combined = np.column_stack((time, E_field))
                    grouped_data[identifier_line][B_field_value] = {
                            "step_fs": step_fs,
                            "start_ps": start_ps,
                            "delay_ms": delay_ms,
                            "data": combined
                    }
                else:
                    grouped_data[identifier_line][B_field_value]["data"] = np.column_stack((grouped_data[identifier_line][B_field_value]["data"], E_field))
            else:
                # Extract data of E-field from the file
                E_field = np.loadtxt(file_path, skiprows=2)

                if step_fs is not None and start_ps is not None:
                    time = start_ps + np.arange(len(E_field)) * step_fs * 1e-3
                    combined = np.column_stack((time, E_field))
                else:
                    combined = np.column_stack((np.zeros_like(E_field), E_field))
                
                # Determine if this is a field-dependent file
                if B_field_dependent:
                    # Extract field value from filename
                    match = re.search(r'_([-+]?\d*\.?\d+)T\.txt$', filename)
                    if not match:
                        print(f"Skipping T file (no match): {filename}")
                        return
                    B_field_value = float(match.group(1))
                else:
                    B_field_value = 0.0

                grouped_data[identifier_line][B_field_value] = {
                    "step_fs": step_fs,
                    "start_ps": start_ps,
                    "delay_ms": delay_ms,
                    "data": combined
                }

        except Exception as e:
            print(f"Error loading T file {filename}: {e}")
    
    # --- Process field-dependent but not saved each scan files ---
    B_field_dependent_not_saved_each_scan_files = [
        f for f in glob.glob(os.path.join(raw_data_dir, '_[0-9]*T.txt'))
        if any(c.isdigit() for c in os.path.basename(f)) and 'T' in os.path.basename(f)
    ]
    for file_path in B_field_dependent_not_saved_each_scan_files:
        extract_data_from_file(file_path, grouped_data=grouped_data, B_field_dependent=True, save_each_scan=False)

    # --- Process save-each-scan files ---
    all_scan_files = glob.glob(os.path.join(raw_data_dir, '*_scan*.txt'))

    save_each_scan_files = []
    B_field_dependent_and_save_each_scan_files = []

    for file_path in all_scan_files:
        filename = os.path.basename(file_path)

        if re.search(r'_[\d.]+T_scan\d+', filename):
            B_field_dependent_and_save_each_scan_files.append(file_path)
        else:
            save_each_scan_files.append(file_path)


    # --- Process: non-field-dependent scan files ---
    for file_path in save_each_scan_files:
        extract_data_from_file(file_path, grouped_data=grouped_data, B_field_dependent=False, save_each_scan=True)

    # --- Process: field-dependent scan files ---
    for file_path in B_field_dependent_and_save_each_scan_files:
        extract_data_from_file(file_path, grouped_data=grouped_data, B_field_dependent=True, save_each_scan=True)


    # --- Process *_Aperture.txt files ---
    aperture_files = glob.glob(os.path.join(raw_data_dir, '*_Aperture.txt'))
    for file_path in aperture_files:
        extract_data_from_file(file_path, grouped_data=grouped_data, B_field_dependent=False)

    # --- Save grouped data ---
    preprocessed_filenames = []
    for identifier, full_data in grouped_data.items():
        output_fits_path = identifier + '.fits'
        os.makedirs(preprocessed_data_dir, exist_ok=True)
        output_fits_path = os.path.join(preprocessed_data_dir, output_fits_path)
        basic_info_and_power = extract_basic_info_and_power(raw_data_dir)
        save_data_to_fits(full_data, output_fits_path, basic_info_and_power)
        preprocessed_filenames.append(output_fits_path)

    return preprocessed_filenames

def save_data_to_fits(data_dict, output_path, basic_info_and_power):
    hdus = [fits.PrimaryHDU()]  # Start with an empty primary HDU
    
    # Stack the data for each B-field value
    data = []
    B_field_values = []

    for B_field_value, info in sorted(data_dict.items()):
        B_field_values.append(B_field_value)
        data.append(info["data"])

    #stacked_data = np.stack(data, axis=0)


    # Input header information
    hdr = fits.Header()

    date, author, power = basic_info_and_power
    hdr['DATE'] = date
    hdr['AUTHOR'] = author
    hdr['POWER'] = power

    info0 = data_dict[B_field_values[0]]
    if info0["step_fs"] is not None:
        hdr['STEPFS'] = info0["step_fs"]
    if info0["start_ps"] is not None:
        hdr['STARTPS'] = info0["start_ps"]
    if info0["delay_ms"] is not None:
        hdr['DELAYMS'] = info0["delay_ms"]

    hdr['N_BFIELD'] = len(B_field_values)
    for i, b in enumerate(B_field_values):
        if b is not None:
            hdr[f'B{i}'] = b

    # Create ImageHDU for each B-field value
    hdu = fits.ImageHDU(data=data, header=hdr, name=f'{B_field_value:.3f}T')
    hdus.append(hdu)
    
    # Write to file
    hdul = fits.HDUList(hdus)
    hdul.writeto(output_path, overwrite=True)
    print(f"Saved to {output_path}")