import numpy as np
class HDU_DATA:
    def __init__(self, data):
        self.times, self.E_field_avgs, self.E_field_stds, self.freqs, self.fft_avg_reals, self.fft_avg_imags, self.fft_stds, self.B_field_values = data
    
    def __add__(self, other):
        if isinstance(other, HDU_DATA):
            e_avg = self.E_field_avgs + other.E_field_avgs
            e_std =  (self.E_field_stds**2 + other.E_field_stds**2)**0.5

            fft_avg_real = self.fft_avg_reals + other.fft_avg_reals
            fft_avg_imag = self.fft_avg_imags + other.fft_avg_imags
            fft_std =  (self.fft_stds**2 + other.fft_stds**2)**0.5

            data_output = self.times, e_avg, e_std, self.freqs, fft_avg_real, fft_avg_imag, fft_std, self.B_field_values
            return HDU_DATA(data= data_output)
    
    def __sub__(self, other):
        if isinstance(other, HDU_DATA):
            e_avg = self.E_field_avgs - other.E_field_avgs
            e_std =  (self.E_field_stds**2 + other.E_field_stds**2)**0.5

            fft_avg_real = self.fft_avg_reals - other.fft_avg_reals
            fft_avg_imag = self.fft_avg_imags - other.fft_avg_imags
            fft_std =  (self.fft_stds**2 + other.fft_stds**2)**0.5

            data_output = self.times, e_avg, e_std, self.freqs, fft_avg_real, fft_avg_imag, fft_std, self.B_field_values
            return HDU_DATA(data= data_output)
    
    def __mul__(self, other):
        if isinstance(other, HDU_DATA):  # element-wise multiplication
            e_avg = self.E_field_avgs * other.E_field_avgs
            e_std =  e_avg * ((self.E_field_stds/self.E_field_avgs)**2 + (other.E_field_stds/other.E_field_avgs)**2)**0.5

            fft_avg_real = self.fft_avg_reals * other.fft_avg_reals - self.fft_avg_imags* other.fft_avg_imags
            fft_avg_imag = self.fft_avg_reals * other.fft_avg_imags + self.fft_avg_imags* other.fft_avg_reals
            fft_mag_avg = (fft_avg_real **2 + fft_avg_imag**2)**0.5
            fft_std =  fft_mag_avg * ((self.fft_stds)**2/(self.fft_avg_reals**2 + self.fft_avg_imags**2) + (other.fft_stds)**2/(self.fft_avg_reals**2 + self.fft_avg_imags**2))**0.5

            data_output = self.times, e_avg, e_std, self.freqs, fft_avg_real, fft_avg_imag, fft_std, self.B_field_values
            return HDU_DATA(data= data_output)
        elif isinstance(other, (int, float)):  # scalar multiplication
            e_avg = self.E_field_avgs * other
            e_std =  self.E_field_stds * np.abs(other)

            fft_avg_real = self.fft_avg_reals * other
            fft_avg_imag = self.fft_avg_imags * other
            fft_std =  self.fft_stds * np.abs(other)

            data_output = self.times, e_avg, e_std, self.freqs, fft_avg_real, fft_avg_imag, fft_std, self.B_field_values
            return HDU_DATA(data= data_output)
    
    def unpack(self):
        return np.array(self.times), np.array(self.E_field_avgs), np.array(self.E_field_stds), np.array(self.freqs), \
            np.array(self.fft_avg_reals), np.array(self.fft_avg_imags), np.array(self.fft_stds), np.array(self.B_field_values)
