import h5py
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import least_squares

def load_waveform(file_path, group_path):
    """Loads time, amplitude, and phase from SpECTRE CCE HDF5 output."""
    with h5py.File(file_path, 'r') as f:
        data = f[group_path][:]
        time = data[:, 0]
        strain_real = data[:, 1]
        strain_imag = data[:, 2]
        
        amplitude = np.sqrt(strain_real**2 + strain_imag**2)
        # Unwrap phase to prevent discontinuity jumps
        phase = np.unwrap(np.angle(strain_real - 1j * strain_imag))
        
    return time, amplitude, phase

def align_waveforms(t_base, amp_base, phase_base, t_mod, amp_mod, phase_mod, t_min, t_max):
    """Aligns modified waveform to baseline using least squares over a shared time window."""
    mask = (t_base >= t_min) & (t_base <= t_max)
    t_align = t_base[mask]
    
    phase_base_align = phase_base[mask]
    
    def residuals(shifts):
        dt, dphi = shifts
        shifted_phase = InterpolatedUnivariateSpline(t_mod, phase_mod)(t_align - dt) + dphi
        return phase_base_align - shifted_phase
        
    # Optimize time and phase shifts
    res = least_squares(residuals, x0=[0.0, 0.0])
    dt_opt, dphi_opt = res.x
    
    # Apply optimal shifts to the full modified waveform
    amp_mod_aligned = InterpolatedUnivariateSpline(t_mod, amp_mod)(t_base - dt_opt)
    phase_mod_aligned = InterpolatedUnivariateSpline(t_mod, phase_mod)(t_base - dt_opt) + dphi_opt
    
    return dt_opt, dphi_opt, amp_mod_aligned, phase_mod_aligned

def calculate_differences(amp_base, phase_base, amp_mod_aligned, phase_mod_aligned):
    """Calculates relative amplitude and absolute phase differences."""
    delta_amp_rel = np.abs(amp_base - amp_mod_aligned) / amp_base
    delta_phase = np.abs(phase_base - phase_mod_aligned)
    
    return delta_amp_rel, delta_phase

if __name__ == "__main__":
    # Example usage for Tier 1 Inspiral Runs (S1, S2, L vs B)
    # Window chosen after initial transients have died away
    T_MIN = 1800.0
    T_MAX = 2200.0
    
    print(f"Post-processing pipeline initialized for alignment window [{T_MIN}M, {T_MAX}M]")
