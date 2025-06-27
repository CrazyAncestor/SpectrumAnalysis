from .ui import input_raw_data, show_fits_info
from SpectrumAnalysis.util import delete_hdu_from_fits, rename_hdu_from_fits, read_stat_hdu, write_data_to_bin_hdu, raw_stat_hdu_name
from SpectrumAnalysis.visualization import plot_statistics, plot_SN_ratio, plot_transmission_spec
from .HDU_DATA import HDU_DATA
import numpy as np

class SpectrumAnalysis:
    def __init__(self, metadata_filename):
        self.metadata_filename = metadata_filename
        input_raw_data(metadata_filename=metadata_filename)
        print('done')
    
    #   Showing basic info
    def show_info(self):
        show_fits_info(fits_path=self.metadata_filename)
    
    #   HDU processing
    def rm_hdu(self, hdu_name):
        delete_hdu_from_fits(self.metadata_filename, hdu_name=hdu_name)
    def mv_hdu(self, hdu_old_name, hdu_new_name):
        rename_hdu_from_fits(self.metadata_filename, hdu_old_name=hdu_old_name, hdu_new_name=hdu_new_name)

    #   Visualization
    def plot_statistics(self, hdu_name, B_field=0., calculate_from_raw_data=True, plot_time_range=None, fft_time_range=None, plot_only_positive_freq=True, freq_range = None, zero_padding_ratio=None, title=None, save_fig=False, save_path=None):
        plot_statistics(fits_file = self.metadata_filename, hdu_name=hdu_name, B_field=B_field, calculate_from_raw_data= calculate_from_raw_data,\
                                plot_time_range=plot_time_range, fft_time_range=fft_time_range, plot_only_positive_freq=plot_only_positive_freq,\
                               freq_range = freq_range, zero_padding_ratio=zero_padding_ratio, title=title, save_fig=save_fig, save_path=save_path)
    def plot_SN_ratio(self, hdu_name, B_field=0., plot_only_positive_freq=True, freq_range = None, title=None, save_fig=False, save_path=None):
        plot_SN_ratio(fits_file = self.metadata_filename, hdu_name=hdu_name, B_field=B_field, plot_only_positive_freq=plot_only_positive_freq,\
                       freq_range = freq_range, title=title, save_fig=save_fig, save_path=save_path)
    def plot_transmission_spec(self, hdu_tr_name, hdu_ref_name, B_field=None, plot_only_positive_freq=True, freq_range = None, title=None, save_fig=False, save_path=None):
        freqs, trans_avg, trans_std = plot_transmission_spec(self.metadata_filename, hdu_tr_name=hdu_tr_name, hdu_ref_name=hdu_ref_name, B_field=B_field, plot_only_positive_freq=plot_only_positive_freq,\
                               freq_range=freq_range, title= title, save_fig=save_fig, save_path=save_path)
        return freqs, trans_avg, trans_std

    #   Calculation on HDUs
    def read_hdu_data(self, hdu_name):
        hdu_raw_name, hdu_stat_name = raw_stat_hdu_name(hdu_name)
        times, E_field_avgs, E_field_stds, freqs, fft_avg_reals, fft_avg_imags, fft_stds, B_field_values, header = read_stat_hdu(self.metadata_filename, hdu_stat_name, B_field='all')
        data = times, E_field_avgs, E_field_stds, freqs, fft_avg_reals, fft_avg_imags, fft_stds, B_field_values
        
        return HDU_DATA(data), header
        
    def create_new_stat_hdu(self, hdu1_name, hdu2_name, new_hdu_name):
        
        HDU_DATA1, header1 = self.read_hdu_data(hdu1_name)
        HDU_DATA2, header2 = self.read_hdu_data(hdu2_name)

        NEW_DATA = HDU_DATA1 + HDU_DATA2
     
        write_data_to_bin_hdu(self.metadata_filename, new_hdu_name, hdu_new_type='STAT', hdu_new_header=header1, stat_data=NEW_DATA.unpack())
    
    def calculate_Ex_Ey(self, hdu1_name, hdu2_name, rot_angle_in_deg, Ex_hdu_name, Ey_hdu_name):
        rot_angle_in_rad = np.deg2rad(rot_angle_in_deg)
        HDU_DATA1, header1 = self.read_hdu_data(hdu1_name)
        HDU_DATA2, header2 = self.read_hdu_data(hdu2_name)

        major_axis = HDU_DATA1 + HDU_DATA2
        minor_axis = HDU_DATA1 - HDU_DATA2

        Ex = major_axis * np.cos(rot_angle_in_rad) + minor_axis * np.sin(rot_angle_in_rad)
        Ey = major_axis * np.sin(rot_angle_in_rad) - minor_axis * np.cos(rot_angle_in_rad)
     
        write_data_to_bin_hdu(self.metadata_filename, 'STAT_' + Ex_hdu_name, hdu_new_type='STAT', hdu_new_header=header1, stat_data=Ex.unpack())
        write_data_to_bin_hdu(self.metadata_filename, 'STAT_' + Ey_hdu_name, hdu_new_type='STAT', hdu_new_header=header1, stat_data=Ey.unpack())