import os
import glob
from astropy.io import fits
import os
import re
from .io import unzip_and_extract_data, clear_directory_files, combine_preprocessed_fits
from .printing import print_color_message, print_help_message

def input_raw_data(metadata_filename='metadata.fits', raw_data_dir='raw_data', preprocessed_data_dir='preprocessed_data'):
    # --- Step 1: Setup ---
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(preprocessed_data_dir, exist_ok=True)
    fits_path = os.path.join('./', metadata_filename)
    create_new_metadata_file(fits_path, metadata_filename)

    # --- Step 2: Open metadata FITS file ---
    with fits.open(fits_path, mode='update') as hdul:
        zip_paths = glob.glob('*.zip')
        if not zip_paths:
            raise ValueError("No zip file found in the current folder!")

        zip_index = 0
        hdu_index = len(hdul)

        def clear_and_reload(zip_idx):
            clear_directory_files(raw_data_dir)
            clear_directory_files(preprocessed_data_dir)
            files = unzip_and_extract_data(zip_paths[zip_idx], raw_data_dir, preprocessed_data_dir)
            prompt_and_combine_fits_with_similar_names(files)
            return files

        # --- Step 3: Initial data load ---
        preprocessed_filenames = clear_and_reload(zip_index)
        print_help_message()

        # --- Step 4: HDU loop ---
        while True:
            hdu_name = ask_user_if_to_create_new_hdu(f"RAWDATA_{hdu_index}", hdul)
            if hdu_name == "null":
                continue
            if hdu_name == "end":
                break

            while True:
                result = add_hdu_with_prompt_message(hdul, preprocessed_filenames, hdu_name)

                if result == "end":
                    break
                elif result == "success":
                    break
                elif result == "next_zip":
                    zip_index = (zip_index + 1) % len(zip_paths)
                    preprocessed_filenames = clear_and_reload(zip_index)
                elif result == "previous_zip":
                    zip_index = (zip_index - 1) % len(zip_paths)
                    preprocessed_filenames = clear_and_reload(zip_index)
                elif result == "help":
                    print_help_message()

            hdu_index += 1
            if result == "end":
                break

    # --- Step 5: Show final result ---
    print("-" * 30)
    show_fits_info(fits_path)

def create_new_metadata_file(fits_path, metadata_filename):
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

def ask_user_if_to_create_new_hdu(hdu_name_default, hdul):
    # Ask user if they want to create a new HDU
    print(f"Do you want to create a new HDU? (y/n): ", flush=True)
    create_new_hdu = input(f"Do you want to create a new HDU? (y/n): ").strip().lower()
    if create_new_hdu != 'y':
        print("Ending the addition of HDU.", flush=True)
        return "end"
    print(f"Enter the name of the new HDU (default: {hdu_name_default}): ", flush=True)
    hdu_name = input(f"Enter the name of the new HDU (default: {hdu_name_default}): ").strip() or hdu_name_default
    if hdu_name in [hdu.name for hdu in hdul]:
        print(f"HDU '{hdu_name}' already exists. Do you want to overwrite it? (y/n): ", flush=True)
        overwrite_hdu = input(f"HDU '{hdu_name}' already exists. Do you want to overwrite it? (y/n): ").strip().lower()
        if overwrite_hdu != 'y':
            print("Choose a new HDU name", flush=True)
            return "null"
        else:
            print(f"Removing existing HDU '{hdu_name}' from the FITS file.", flush=True)
            del hdul[hdu_name]

    return hdu_name

def add_hdu_with_prompt_message(hdul, preprocessed_filenames, hdu_name):
    """
    Prompts the user to select a FITS file, stacks its 2D HDUs into a 3D array,
    and appends it to the specified FITS file as a new ImageHDU.

    Parameters:
        hdul (HDUList): The FITS file to which the new HDU will be added.
        preprocessed_filenames (list of str): List of full paths to candidate FITS files.
        hdu_name (str): Name to assign to the new HDU (e.g., 'RAWDATA_1').
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
        new_hdu = fits.ImageHDU(header=hdul_input[1].header ,data=hdul_input[1].data, name=hdu_name)
        
        # Set the HDU_TYPE to be RAW_DATA
        new_hdu.header['HDU_TYPE'] = 'RAW_DATA'

    # Append the new HDU to the original FITS file
    hdul.append(new_hdu.copy())
    print(f"HDU '{hdu_name}' added successfully.", flush=True)
    return "success"

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
            base = os.path.splitext(os.path.basename(fname))[0]
            base_names.add(base)
    for base in base_names:
        matching_files = [
            f for f in filenames
            if (
                f.endswith('.fits') and (
                    os.path.basename(f) == f"{base}.fits" or
                    re.match(rf"^{re.escape(base)}_[-+]?\d*\.?\d+T\.fits$", os.path.basename(f))
                )
            )
        ]
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