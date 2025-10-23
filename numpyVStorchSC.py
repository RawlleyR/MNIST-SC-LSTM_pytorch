# This script compares the outputs of stochastic computing functions
# implemented in NumPy and PyTorch.
# It assumes the following files exist in the same directory:
# - stochastic.py (containing the NumPy functions)
# - stochastic_torch.py (containing the PyTorch functions)

import torch
import numpy as np

# Import the files with aliases to avoid function name conflicts
import stochastic as scnp
import stochastic_torch as scpt

if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # --- SETUP FOR COMPARISON ---
    # We must set seeds to ensure the random bit streams are identical for both versions.
    # This is the most crucial step for a fair side-by-side comparison.
    seed_value = 42
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    
    # --- TEST CASES ---
    
    val_to_test = np.array([[0.5, 0.7, 0.1],[0.8, 0.9, 1]])
    
    # NumPy version
    np_stream = scnp.bip(val_to_test)
    np_stream_val = scnp.bsn_actual_value(np_stream)
    np_tanh_stream = scnp.tanh_activation(np_stream)
    np_tanh_val = scnp.bsn_actual_value(np_tanh_stream)

    # PyTorch version
    pt_stream = scpt.bip(torch.from_numpy(val_to_test).to(device)).to('cpu').numpy()
    pt_stream_val = scpt.bsn_actual_value(torch.from_numpy(pt_stream).to(device))
    pt_tanh_stream = scpt.tanh_activation(torch.from_numpy(pt_stream).to(device))
    pt_tanh_val = scpt.bsn_actual_value(pt_tanh_stream)
    
    print(f"Original value: {val_to_test}")
    print(f"NumPy bip stream value: {np_stream_val}")
    print(f"PyTorch bip stream value: {pt_stream_val}")
    print(f"NumPy tanh value: {np_tanh_val}")
    print(f"PyTorch tanh value: {pt_tanh_val}")