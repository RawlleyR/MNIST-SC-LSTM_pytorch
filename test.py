# from stochastic import *
# from apc import bsn_apc_2in_sum

# a = np.array([0.3, 0.7])
# b = np.array([0.4, 0.5])

# a_bsn = bip(a)
# b_bsn = bip(b)

# bsn_sum = bsn_apc_2in_sum(a_bsn, b_bsn)

# print('actual sum: ', a+b)
# print('bsn sum: ', bsn_actual_value(bsn_sum))

# import torch
# import pandas as pd

# # Load .pt file
# pt_file_path = 'cw_targeted_9_adv_500samples_tensorattacks_batch1-5(63.8%_1000_0_10000).pt'
# data = torch.load(pt_file_path, map_location='cpu')

# # Make sure all values are tensors and same shape
# for key, value in data.items():
#     if not torch.is_tensor(value):
#         raise ValueError(f"Value for key '{key}' is not a tensor.")
        
# # Convert each tensor to a list
# data_lists = {key: value.tolist() for key, value in data.items()}

# # Create DataFrame where keys become column headers
# df = pd.DataFrame(data_lists)

# # Save to Excel
# df.to_excel('targeted_attack_batch1-5.xlsx', index=False)
# print("Saved clean Excel file with expanded tensors.")

# code to generate tensor array of bestcase target classes from excel file to feed into the attack
import pandas as pd
import torch

# Load Excel file
df = pd.read_excel("cw_attack_summary_params-100_0_1000.xlsx")

# Extract the "Best Target (min L2)" column
best_targets = df["Best Target (min L2)"]

# Replace 'None' entries with -1 (or any placeholder you prefer)
best_targets = best_targets.fillna(8).astype(int)

# Convert to tensor (dtype long for class indices)
best_target_tensor = torch.tensor(best_targets.values, dtype=torch.long)
best_target_tensor[465] = 3

print(best_target_tensor.shape)   # should match number of images
# print(best_target_tensor[:20])    # preview first 20 entries

# Save tensor if needed
torch.save(best_target_tensor, "bestcase_target_classes_params-100_0_1000.pt")
