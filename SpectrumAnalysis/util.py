from astropy.io import fits
import numpy as np

def find_B_field_idx(header, B_field):
    b_field_list = [header[f'B{i}'] for i in range(0, header['N_BFIELD'])]
    try:
        B_field_idx = b_field_list.index(B_field)
    except ValueError:
        raise ValueError(f"B-field {B_field} not found in the list.")
    return B_field_idx

def give_B_field_values(header):
    return [header[f'B{i}'] for i in range(0, header['N_BFIELD'])]


def confirm_whether_hdu_exist_and_if_overwrite(fits_file, hdu_id, allow_overwrite=False):
    with fits.open(fits_file) as hdul:
        if hdu_id not in hdul:
            print(f"HDU {hdu_id} not found in {fits_file}.", flush=True)
            hdu_exist = False
        else:
            print(f"HDU {hdu_id} found in {fits_file}.", flush=True)
            hdu_exist = True
        
    overwrite_or_not = False
    if hdu_exist and allow_overwrite==True:
        print(f"Do you want to overwrite current {hdu_id} from {fits_file}? (y/n): ", flush=True)
        delete_hdu = input(f"Do you want to remove current {hdu_id} from {fits_file}? (y/n): ")
        if delete_hdu.lower() == 'y':
            print(f"Removing {hdu_id} from {fits_file}.")
            delete_hdu_from_fits(fits_file, hdu_id)
            overwrite_or_not = True
        else:
            print(f"Keeping {hdu_id} in {fits_file}.")
    
    return hdu_exist, overwrite_or_not

def read_image_hdu(fits_file, hdu_id):
    with fits.open(fits_file) as hdul:
        hdu = hdul[hdu_id]
        header = hdu.header.copy()
        data = hdu.data.copy()
        return header, np.array(data)

def raw_stat_hdu_name(hdu_id):
    if hdu_id[:5] == 'STAT_':
        hdu_stat_id = hdu_id
    else:
        hdu_stat_id = 'STAT_' + hdu_id
    hdu_raw_data_id = hdu_stat_id[5:]
    return hdu_raw_data_id, hdu_stat_id

def read_stat_hdu(fits_file, hdu_id, B_field):
    hdu_raw_data_id, hdu_stat_id = raw_stat_hdu_name(hdu_id)

    with fits.open(fits_file) as hdul:
        data = hdul[hdu_stat_id].data
        header = hdul[hdu_stat_id].header

        time_N = len(data[0][0])/header['N_BFIELD']
        freq_N = len(data[3][0])/header['N_BFIELD']

        if B_field == 'all':
            times = np.array(data[0][0]).reshape(header['N_BFIELD'], int(time_N))
            E_field_avgs = np.array(data[1][0]).reshape(header['N_BFIELD'], int(time_N))
            E_field_stds = np.array(data[2][0]).reshape(header['N_BFIELD'], int(time_N))
            freqs = np.array(data[3][0]).reshape(header['N_BFIELD'], int(freq_N))
            fft_avgs = np.array(data[4][0]).reshape(header['N_BFIELD'], int(freq_N))
            fft_stds = np.array(data[5][0]).reshape(header['N_BFIELD'], int(freq_N))

            B_field_values = give_B_field_values(header.copy())
        else:
            B_field_idx = find_B_field_idx(header, B_field)
            time_idx = slice(B_field_idx * int(time_N), (B_field_idx + 1) * int(time_N))
            freq_idx = slice(B_field_idx * int(freq_N), (B_field_idx + 1) * int(freq_N))

            times = np.array(data[0][0][time_idx])[np.newaxis, :]
            E_field_avgs = np.array(data[1][0][time_idx])[np.newaxis, :]
            E_field_stds = np.array(data[2][0][time_idx])[np.newaxis, :]
            freqs = np.array(data[3][0][freq_idx])[np.newaxis, :]
            fft_avgs = np.array(data[4][0][freq_idx])[np.newaxis, :]
            fft_stds = np.array(data[5][0][freq_idx])[np.newaxis, :]

            B_field_values = [give_B_field_values(header.copy())[B_field_idx]]

        

    return times, E_field_avgs, E_field_stds, freqs, fft_avgs, fft_stds, B_field_values

def write_data_to_bin_hdu(fits_file, hdu_new_id, hdu_new_type, hdu_new_header, stat_data):
    with fits.open(fits_file, mode='update') as hdul:

        # Flatten the arrays
        stat_arrs = [arr.flatten().astype(np.float32) for arr in stat_data]

        # Convert to a format that fits understands: an object array
        col = fits.Column(name=hdu_new_id, format='PE()', array=np.array(stat_arrs, dtype=object))

        # Make a binary table
        hdu_new = fits.BinTableHDU.from_columns([col])
        hdu_new.header = hdu_new_header.copy()
        hdu_new.header['EXTNAME'] = hdu_new_id
        hdu_new.header['HDU_TYPE'] = hdu_new_type
        print(f"Adding {hdu_new.name} to the FITS file.")
        print(hdu_new.header)

        # Append the new HDU to the original HDU list
        hdul.append(hdu_new.copy())

def delete_hdu_from_fits(fits_file, hdu_name, output_file=None):
    """
    Deletes an HDU from a FITS file by its name.

    Parameters:
        fits_file (str): Path to the input FITS file.
        hdu_name (str): Name of the HDU to delete.
        output_file (str, optional): Path to save the modified FITS file. 
                                     If None, the original file will be overwritten.

    Returns:
        None
    """
    # Open the FITS file
    with fits.open(fits_file, mode='update') as hdul:
        # Find the HDU by name
        hdu_index = None
        for i, hdu in enumerate(hdul):
            if hdu.name == hdu_name:
                hdu_index = i
                break
        
        # Check if the HDU was found
        if hdu_index is None:
            raise ValueError(f"HDU with name '{hdu_name}' not found in the FITS file.")
        
        # Delete the specified HDU
        del hdul[hdu_index]
        
        # Save changes to the file
        if output_file:
            hdul.writeto(output_file, overwrite=True)
        else:
            hdul.flush()

def rename_hdu_from_fits(fits_file, hdu_old_name, hdu_new_name):
    
    # Open the FITS file
    with fits.open(fits_file, mode='update') as hdul:
        # Find the HDU by name
        hdu_index = None
        for i, hdu in enumerate(hdul):
            if hdu.name == hdu_old_name:
                hdu.header['EXTNAME']= hdu_new_name
                break
        
        # Check if the HDU was found
        if hdu_index is None:
            raise ValueError(f"HDU with name '{hdu_old_name}' not found in the FITS file.")
