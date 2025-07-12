# from stochastic import *
# from apc import bsn_apc_2in_sum

# a = np.array([0.3, 0.7])
# b = np.array([0.4, 0.5])

# a_bsn = bip(a)
# b_bsn = bip(b)

# bsn_sum = bsn_apc_2in_sum(a_bsn, b_bsn)

# print('actual sum: ', a+b)
# print('bsn sum: ', bsn_actual_value(bsn_sum))

import torch
import pandas as pd

# Load .pt file
pt_file_path = 'cw_adversarial_500samples_tensorattacks_batch46-50(16%_10_1).pt'
data = torch.load(pt_file_path, map_location='cpu')

# Make sure all values are tensors and same shape
for key, value in data.items():
    if not torch.is_tensor(value):
        raise ValueError(f"Value for key '{key}' is not a tensor.")
        
# Convert each tensor to a list
data_lists = {key: value.tolist() for key, value in data.items()}

# Create DataFrame where keys become column headers
df = pd.DataFrame(data_lists)

# Save to Excel
df.to_excel('batch46-50_output.xlsx', index=False)
print("Saved clean Excel file with expanded tensors.")
