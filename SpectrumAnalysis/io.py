import os
import glob
import re
import numpy as np
from astropy.io import fits
from collections import defaultdict
import zipfile
import os
import shutil
import time

def clear_directory_files(directory):
    """
    Clear all files in the specified directory.
    
    Args:
        directory (str): Path to the directory to clear.
    """
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def unzip_and_extract_data(zip_path, raw_data_dir, preprocessed_data_dir):
    unzip_file(zip_path, extract_to=raw_data_dir)
    preprocessed_filenames = extract_data_from_files(raw_data_dir, preprocessed_data_dir=preprocessed_data_dir)
    
    return preprocessed_filenames

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

def extract_data_from_files(raw_data_dir, preprocessed_data_dir=None):
    grouped_data = defaultdict(dict)
    preprocessed_filenames = []

    def match_files(pattern, condition=None):
        files = glob.glob(os.path.join(raw_data_dir, pattern))
        return [f for f in files if condition is None or condition(os.path.basename(f))]

    # --- Categorize files ---
    all_txt_files = match_files('*.txt')
    log_files = match_files('*_log.txt')
    info_files = match_files('*_info.txt')

    B_field_dependent_not_each_scan = match_files('*_*T.txt',
        lambda name: re.search(r'_[-+]?\d*\.?\d+T\.txt$', name))

    B_field_dependent_each_scan = match_files('*_scan*.txt',
        lambda name: re.search(r'_[-+]?\d*\.?\d+T_scan\d+', name))

    all_scan_files = match_files('*_scan*.txt')
    not_field_dependent_scans = list(set(all_scan_files) - set(B_field_dependent_each_scan))

    other_files = list(set(all_txt_files) -
                       set(B_field_dependent_each_scan) -
                       set(B_field_dependent_not_each_scan) -
                       set(not_field_dependent_scans) -
                       set(log_files) -
                       set(info_files))

    # --- Process categorized files ---
    categorized_inputs = [
        (B_field_dependent_not_each_scan, True, False),
        (not_field_dependent_scans, False, True),
        (B_field_dependent_each_scan, True, True),
        (other_files, False, False),
    ]

    for file_list, is_field_dependent, is_scan in categorized_inputs:
        for file_path in file_list:
            extract_data_from_file(
                file_path,
                grouped_data=grouped_data,
                B_field_dependent=is_field_dependent,
                save_each_scan=is_scan
            )

    # --- Save processed data ---
    if preprocessed_data_dir:
        os.makedirs(preprocessed_data_dir, exist_ok=True)

        for identifier, full_data in grouped_data.items():
            output_fits_path = os.path.join(preprocessed_data_dir, f"{identifier}.fits")
            basic_info_and_power = extract_basic_info_and_power(raw_data_dir)
            save_data_to_fits(full_data, output_fits_path, basic_info_and_power, identifier)
            preprocessed_filenames.append(output_fits_path)

    return preprocessed_filenames


def extract_data_from_file(file_path, grouped_data, B_field_dependent, save_each_scan=False):
    filename = os.path.basename(file_path)

    try:
        with open(file_path, 'r') as f:
            header_line = f.readline().strip()
            identifier_line = f.readline().strip()

        header_info = parse_header_line(header_line)
        if not header_info:
            print(f"Warning: couldn't parse header in {filename}. Skipping this file.")
            return

        data = np.loadtxt(file_path, skiprows=2)
        time = None
        B_field_value = 0.0  # default

        # Handle B-field if applicable
        if B_field_dependent:
            match = re.search(r'_([-+]?\d*\.?\d+)T', filename)
            if not match:
                print(f"Skipping T file (no match): {filename}")
                return
            B_field_value = float(match.group(1))

        # Determine time and E-field
        if save_each_scan:
            E_field = data[:, 1]
            time = data[:, 0]
        else:
            E_field = data
            if header_info['step_fs'] is not None:
                time = header_info['step_fs'] + np.arange(len(E_field)) * header_info['step_fs'] * 1e-3
            else:
                time = np.zeros_like(E_field)

        combined = np.column_stack((time, E_field))

        # Insert into grouped_data
        if save_each_scan and B_field_value in grouped_data[identifier_line]:
            # Stack E-field column onto existing data
            grouped_data[identifier_line][B_field_value]["data"] = np.column_stack((
                grouped_data[identifier_line][B_field_value]["data"], E_field))
        else:
            grouped_data[identifier_line][B_field_value] = {
                "step_fs": header_info['step_fs'],
                "start_ps": header_info['start_ps'],
                "delay_ms": header_info['delay_ms'],
                "data": combined
            }

    except Exception as e:
        print(f"Error loading T file {filename}: {e}")


def combine_arrays_of_differnt_shape(arrays, filename):
    """
    Combines a list of NumPy arrays along axis 0.
    If arrays differ in size along axis 2, asks the user whether to truncate.
    
    Parameters:
        arrays (List[np.ndarray]): List of NumPy arrays to combine.
    
    Returns:
        np.ndarray: Combined array along axis 0, or None if user declines truncation.
    """
    print(f"Processing file {filename}",flush=True)
    # Get shapes along axis 2
    axis2_lengths = [arr.shape[2] for arr in arrays if arr.ndim >= 3]

    if len(set(axis2_lengths)) > 1:
        print("Arrays differ in shape along axis 2:",flush=True)
        for i, arr in enumerate(arrays):
            print(f"  Array {i}: shape = {arr.shape}")
        
        print("The two data have different shapes. Do you want to truncate all arrays to the smallest length on axis 2? (y/n): ",flush=True)
        answer = input("Do you want to truncate all arrays to the smallest length on axis 2? (y/n): ").strip().lower()
        
        if answer != 'y':
            print("Aborting combination.")
            return None
        
        min_len = min(axis2_lengths)
        print(f"Truncating all arrays to axis 2 size: {min_len}",flush=True)
        arrays = [arr[:, :, :min_len] if arr.shape[2] > min_len else arr for arr in arrays]

    # Combine on axis 0
    return np.concatenate(arrays, axis=0)

def combine_preprocessed_fits(fits1, fits2):
    B_fields_new = []
    with fits.open(fits1, mode='readonly') as hdul:
        hdu = hdul[1].copy()
        header = hdu.header.copy()
        data_to_be_combined = hdu.data.copy()
        n_increase = header['N_BFIELD']
        for i in range(n_increase):
            B_fields_new.append(header[f'B{i}'])
    time.sleep(0.1)
    with fits.open(fits2, mode='update') as hdul:
        hdu = hdul[1]

        # update header
        header = hdu.header
        n_old = header['N_BFIELD']
        for i in range(n_increase):
            header[f'B{i + n_old}'] = B_fields_new[i]
        header['N_BFIELD'] = n_old + n_increase

        # update data
        data_old = hdu.data.copy()
        data_new = combine_arrays_of_differnt_shape(arrays=[data_old, data_to_be_combined], filename=fits2)
        hdu.data = data_new
    time.sleep(0.1)
    os.remove(fits1)

def parse_header_line(header_line):
    """
    Parses a header line to extract step_fs, points, start_ps, and delay_ms.
    
    Args:
        header_line (str): The header line to parse.
    
    Returns:
        dict: A dictionary with extracted values if the pattern matches.
        bool: False if the pattern does not match.
    """
    # Define the regex pattern
    pattern = r'step(\d+)fs_(\d+)pnts_from([-+]?\d+)ps_delay(\d+)ms'
    
    # Match the pattern
    match = re.search(pattern, header_line)
    if match:
        # Extract values
        step_fs = int(match.group(1))
        points = int(match.group(2))
        start_ps = int(match.group(3))
        delay_ms = int(match.group(4))
        
        # Return the extracted values as a dictionary
        return {
            "step_fs": step_fs,
            "points": points,
            "start_ps": start_ps,
            "delay_ms": delay_ms
        }
    else:
        # Return False if the pattern does not match
        return False

def save_data_to_fits(data_dict, output_path, basic_info_and_power, identifier):
    hdus = [fits.PrimaryHDU()]  # Start with an empty primary HDU
    
    # Stack the data for each B-field value
    arrs = []
    B_field_values = []

    for B_field_value, info in sorted(data_dict.items()):
        B_field_values.append(B_field_value)
        arr = np.array(info["data"])
        arr = arr[np.newaxis,:]
        arrs.append(arr)

    data = combine_arrays_of_differnt_shape(arrs, output_path)

    # Input header information
    hdr = fits.Header()

    date, author, power, place, geometry = basic_info_and_power
    hdr['DATE'] = date
    hdr['AUTHOR'] = author
    hdr['POWER'] = power
    hdr['PLACE'] = place
    hdr['GEOMETRY'] = geometry

    info0 = data_dict[B_field_values[0]]
    if info0["step_fs"] is not None:
        hdr['STEPFS'] = info0["step_fs"]
    if info0["start_ps"] is not None:
        hdr['STARTPS'] = info0["start_ps"]
    if info0["delay_ms"] is not None:
        hdr['DELAYMS'] = info0["delay_ms"]

    hdr['FILENAME'] = identifier

    hdr['N_BFIELD'] = len(B_field_values)
    for i, b in enumerate(B_field_values):
        if b is not None:
            hdr[f'B{i}'] = b

    # Create ImageHDU to store the data
    hdu = fits.ImageHDU(data=data, header=hdr, name=identifier.upper())
    hdus.append(hdu)
    
    # Write to file
    hdul = fits.HDUList(hdus)
    hdul.writeto(output_path, overwrite=True)

def extract_basic_info_and_power(raw_data_dir):
    # Step 2: Extract basic_info.txt
    basic_info_path = os.path.join(raw_data_dir, 'basic_info.txt')
    if os.path.exists(basic_info_path):
        with open(basic_info_path, 'r') as f:
            date = f.readline().strip()
            author = f.readline().strip()
            place = f.readline().strip()
            geometry = f.readline().strip()
    else:
        date = 'UNKNOWN'
        author = 'UNKNOWN'
        place = 'UNKNOWN'
        geometry = 'UNKNOWN'

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
    return date, author, power, place, geometry
