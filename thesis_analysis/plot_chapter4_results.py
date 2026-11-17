import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

# Configure matplotlib for publication-quality academic plots
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 1.5,
})

def plot_scale_factors(time, s_gamma, s_mu, output_dir):
    """Plots the evolution of the damping and gauge scale factors (Section 4.4)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time, s_gamma, label=r'$s_\gamma(t)$ (Gauge)', color='blue')
    ax.plot(time, s_mu, label=r'$s_\mu(t)$ (Damping)', color='red', linestyle='--')
    
    ax.set_xlabel(r'Time $(M)$')
    ax.set_ylabel('Scale Factor')
    ax.set_title('Dynamic Scale Factor Evolution (Run L)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_scale_factors.pdf'))
    plt.close()

def plot_constraint_energy(time, ce_baseline, ce_feedback, output_dir):
    """Plots the constraint energy comparison between baseline and feedback runs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(time, ce_baseline, label='Baseline (Run B)', color='black', alpha=0.7)
    ax.semilogy(time, ce_feedback, label='Feedback Loop (Run L)', color='green')
    ax.axhline(y=1.0e-8, color='gray', linestyle=':', label=r'Threshold ($c_{hi}$)')
    
    ax.set_xlabel(r'Time $(M)$')
    ax.set_ylabel('Constraint Energy Norm')
    ax.set_title('Constraint Energy Optimization')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_constraint_energy.pdf'))
    plt.close()

if __name__ == "__main__":
    OUT_DIR = "plots"
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Visualization pipeline ready. Publication plots will be saved to ./{OUT_DIR}/")
