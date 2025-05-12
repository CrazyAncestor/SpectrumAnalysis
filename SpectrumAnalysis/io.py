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



def prompt_and_combine_fits_with_similar_names(filenames):
    """
    Given a list of .fits filenames, find all related files (same base name),
    and ask the user if they want to combine the data.

    Parameters:
        filenames (List[str]): A list of full .fits file paths.
    """

    # Extract unique base names (e.g., "abcd" from "abcd.fits" and "abcd_0T.fits")
    base_names = set()
    for fname in filenames:
        if fname.endswith('.fits'):
            just_name = os.path.basename(fname)
            base = just_name.split('.')[0].split('_')[0]
            base_names.add(base)

    for base in base_names:
        matching_files = [f for f in filenames if os.path.basename(f).startswith(base) and f.endswith('.fits')]
        if not matching_files:
            continue

        shortest_file = min(matching_files, key=lambda f: len(os.path.basename(f)))
        matching_files.remove(shortest_file)
        matching_files.insert(0, shortest_file)

        if len(matching_files) > 1:
            print(f"\nFound related FITS files for base '{base}':", flush=True)
            for f in matching_files:
                print(f"  {os.path.basename(f)}")

            answer = input("Do you want to combine data from these files? (y/n): ").strip().lower()
            if answer == 'y':
                print(f"Combining data for: {', '.join([os.path.basename(f) for f in matching_files])}",flush=True)
                for f in matching_files[1:]:
                    combine_preprocessed_fits(f, shortest_file)
                    filenames.remove(f)
            else:
                print("Skipping.")

def process_data_zip(metadata_filename ='metadata.fits',raw_data_dir='raw_data',preprocessed_data_dir='preprocessed_data'):

    # Step 5: Save the FITS file
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(preprocessed_data_dir, exist_ok=True)
    fits_path = os.path.join('./', metadata_filename)

    if not(os.path.exists(fits_path)):
        # Create a new FITS file with a PrimaryHDU
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['FILENAME'] = metadata_filename

        print("1. Author of this metadata: ", flush=True)
        primary_hdu.header['AUTHOR'] = input("Enter the author of this metadata: ")
        print("2. Date of writing this data analysis: ", flush=True)
        primary_hdu.header['DATE'] = input("Enter the date of this metadata: ")
        print("3. Brief description of this project: ", flush=True)
        primary_hdu.header['PROJECT_GOAL'] = input("Enter the goal of this project: ")
        
        hdul = fits.HDUList([primary_hdu])
        hdul.writeto(fits_path, overwrite=True)

    with fits.open(fits_path, mode='update') as hdul:

        def unzip_and_extract_data(zip_path, raw_data_dir, preprocessed_data_dir):
            # Step 1: Unzip
            unzip_file(zip_path, extract_to=raw_data_dir)

            # Step 6: Continue processing other data
            preprocessed_filenames = extract_data_from_files(raw_data_dir, preprocessed_data_dir=preprocessed_data_dir)
            return preprocessed_filenames
    
        # Step 8: Add HDU to the FITS file
        hdu_index = 1
        zip_index = 0

        hdu_name_prefix = ["REF", "SAMPLE"]
        zip_paths = glob.glob('*.zip')
        if len(zip_paths)==0:
            raise ValueError("No zip file found in the current folder!")
        preprocessed_filenames = unzip_and_extract_data(zip_paths[zip_index], raw_data_dir, preprocessed_data_dir)
        prompt_and_combine_fits_with_similar_names(preprocessed_filenames)

        print_help_message()
        while True:
            result = "null"
            for i in range(2):
                hdu_name = f"{hdu_name_prefix[i]}{hdu_index}"
                
                if hdu_name in [hdu.name for hdu in hdul]:
                    print(f"HDU '{hdu_name}' already exists. Skipping...", flush=True)
                    continue
    
                while True:
                    result = add_hdu_with_prompt_message(hdul, preprocessed_filenames, hdu_name)

                    if result == "end":
                        # Clear the raw data directory files
                        clear_directory_files(raw_data_dir)
                        
                        # Clear the preprocessed data directory files
                        clear_directory_files(preprocessed_data_dir)
                        break
                    if result == "skip":
                        break
                    if result == "next_zip":
                        # Clear the raw data directory files
                        clear_directory_files(raw_data_dir)
                        
                        # Clear the preprocessed data directory files
                        clear_directory_files(preprocessed_data_dir)
                        zip_index += 1
                        if zip_index>= len(zip_paths):
                            zip_index = 0
                        preprocessed_filenames = unzip_and_extract_data(zip_paths[zip_index], raw_data_dir, preprocessed_data_dir)
                    if result == "previous_zip":
                        # Clear the raw data directory files
                        clear_directory_files(raw_data_dir)
                        
                        # Clear the preprocessed data directory files
                        clear_directory_files(preprocessed_data_dir)
                        zip_index += -1
                        if zip_index< 0 :
                            zip_index = len(zip_paths)-1
                        preprocessed_filenames = unzip_and_extract_data(zip_paths[zip_index], raw_data_dir, preprocessed_data_dir)
                    if result == "success":
                        break
                    if result == "help":
                        print_help_message()
                        continue
                
                if result == "end":
                    break

            hdu_index += 1
            if result == "end":
                break
    
    # Show the FITS file information
    print("-" * 30)
    show_fits_info(fits_path)

def print_help_message():
    print_color_message(f"Press the index number to choose the file to read into the hdu", color_code=33)# yellow
    print_color_message(f"Press 's' to skip the addition of HDU", color_code=33)
    print_color_message(f"Press 'e' to end the addition of HDU", color_code=33)
    print_color_message(f"Press 'n' to go to the next zip file", color_code=33)
    print_color_message(f"Press 'p' to go to the previous zip file", color_code=33)

def print_color_message(text, color_code):
    print(f"\033[{color_code}m{text}\033[0m", flush=True)

def add_hdu_with_prompt_message(hdul, preprocessed_filenames, hdu_name="REF1"):
    """
    Prompts the user to select a FITS file, stacks its 2D HDUs into a 3D array,
    and appends it to the specified FITS file as a new ImageHDU.

    Parameters:
        hdul (HDUList): The FITS file to which the new HDU will be added.
        preprocessed_filenames (list of str): List of full paths to candidate FITS files.
        hdu_name (str): Name to assign to the new HDU (e.g., 'REF1', 'SAMPLE1').
    """
    # Print the list of available files
    print("-" * 30)
    print_color_message(f"Please select the FITS file to add as HDU '{hdu_name}' (press 'h' for help):", color_code=35) # magenta
    print_color_message(f"Available files:", color_code=34)
    
    for i, full_path in enumerate(preprocessed_filenames):
        identifer = os.path.basename(full_path)
        print_color_message(f"{i}: {identifer}", color_code=32) # green

    # Validate user input
    while True:

        input_message = input("Enter the index of the file: ")

        if input_message.lower() == 's':
            print("Skipping the addition of HDU.", flush=True)
            return "skip"
        if input_message.lower() == 'e':
            print("Ending the addition of HDU.", flush=True)
            return "end"
        if input_message.lower() == 'n':
            print("Go to next zip file.", flush=True)
            return "next_zip"
        if input_message.lower() == 'p':
            print("Go to previous zip file.", flush=True)
            return "previous_zip"
        if input_message.lower() == 'h':
            print("Help message", flush=True)
            return "help"
        try:
            index = int(input_message)
            if 0 <= index < len(preprocessed_filenames):
                selected_filename = preprocessed_filenames[int(input_message)]
                print(f"You selected: {selected_filename}", flush=True)
                break
            else:
                print("Index out of range. Try again.", flush=True)
        except ValueError:
            print("Invalid input. Please enter a number.", flush=True)


    # Open and process the selected FITS file
    with fits.open(selected_filename) as hdul_input:
        # Create and append the new ImageHDU
        new_hdu = fits.ImageHDU(header=hdul_input[1].header ,data=hdul_input[1].data, name=hdu_name.upper())
        
        # Set the HDU_TYPE to be RAW_DATA
        new_hdu.header['HDU_TYPE'] = 'RAW_DATA'

    # Append the new HDU to the original FITS file
    hdul.append(new_hdu.copy())
    print(f"HDU '{hdu_name.upper()}' added successfully.", flush=True)
    return "success"
    
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
                print(f"Warning: couldn't parse header in {filename}. Skipping this file")
                step_fs = start_ps = delay_ms = None
                return

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
    
    all_txt_files = glob.glob(os.path.join(raw_data_dir, '*.txt'))
    all_scan_files = glob.glob(os.path.join(raw_data_dir, '*_scan*.txt'))
    info_files = glob.glob(os.path.join(raw_data_dir, '*_info.txt'))
    log_files = glob.glob(os.path.join(raw_data_dir, '*_log.txt'))

    B_field_dependent_not_saved_each_scan_files = [
        f for f in glob.glob(os.path.join(raw_data_dir, '*_*T.txt'))
        if re.search(r'_[-+]?\d*\.?\d+T\.txt$', os.path.basename(f))
    ]
    B_field_dependent_saved_each_scan_files = [
        f for f in glob.glob(os.path.join(raw_data_dir, '*_scan*.txt'))
        if re.search(r'_[-+]?\d*\.?\d+T_scan\d+', os.path.basename(f))
    ]
    all_scan__files_not_field_dependent = list(set(all_scan_files) - \
                                            set(B_field_dependent_saved_each_scan_files))
    other_files = list(set(all_txt_files) - \
                       set(B_field_dependent_saved_each_scan_files) - \
                       set(B_field_dependent_not_saved_each_scan_files) - \
                       set(all_scan__files_not_field_dependent) - \
                       set(log_files) - set(info_files))

    # --- Process field-dependent but not saved each scan files ---
    for file_path in B_field_dependent_not_saved_each_scan_files:
        extract_data_from_file(file_path, grouped_data=grouped_data, B_field_dependent=True, save_each_scan=False)

    # --- Process: non-field-dependent scan files ---
    for file_path in all_scan__files_not_field_dependent:
        extract_data_from_file(file_path, grouped_data=grouped_data, B_field_dependent=False, save_each_scan=True)

    # --- Process: field-dependent scan files ---
    for file_path in B_field_dependent_saved_each_scan_files:
        extract_data_from_file(file_path, grouped_data=grouped_data, B_field_dependent=True, save_each_scan=True)

    # --- Process other files ---
    for file_path in other_files:
        extract_data_from_file(file_path, grouped_data=grouped_data, B_field_dependent=False, save_each_scan=False)

    # --- Save grouped data ---
    preprocessed_filenames = []
    for identifier, full_data in grouped_data.items():

        # Create a unique filename for each identifier
        output_fits_path = identifier + '.fits'
        os.makedirs(preprocessed_data_dir, exist_ok=True)
        output_fits_path = os.path.join(preprocessed_data_dir, output_fits_path)

        # Extract basic info and power
        basic_info_and_power = extract_basic_info_and_power(raw_data_dir)

        # Save the data to a FITS file
        save_data_to_fits(full_data, output_fits_path, basic_info_and_power, identifier)
        preprocessed_filenames.append(output_fits_path)

    return preprocessed_filenames

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

def show_fits_info(fits_path):
    """
    Prints the information of a FITS file.
    
    Args:
        fits_path (str): Path to the FITS file.
    """
    with fits.open(fits_path) as hdul:
        hdul.info()
        print('Primary HDU Info:')
        print(hdul[0].header)
        for hdu in hdul[1:]:
            print("-" * 30)
            print(f"HDU name: {hdu.name}")
            print(hdu.header['FILENAME'])
            print(f"Data Date: {hdu.header['DATE']}")
            print(f"Data shape: {hdu.data.shape}")
            print(f"Magnetic field values: {hdu.header['N_BFIELD']}")
            for i in range(hdu.header['N_BFIELD']):
                print(f"B{i}: {hdu.header[f'B{i}']}")