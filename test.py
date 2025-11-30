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

# # code to generate tensor array of bestcase target classes from excel file to feed into the attack
# import pandas as pd
# import torch

# # Load Excel file
# df = pd.read_excel("cw_attack_summary_params-100_0_1000.xlsx")

# # Extract the "Best Target (min L2)" column
# best_targets = df["Best Target (min L2)"]

# # Replace 'None' entries with -1 (or any placeholder you prefer)
# best_targets = best_targets.fillna(8).astype(int)

# # Convert to tensor (dtype long for class indices)
# best_target_tensor = torch.tensor(best_targets.values, dtype=torch.long)
# best_target_tensor[465] = 3

# print(best_target_tensor.shape)   # should match number of images
# # print(best_target_tensor[:20])    # preview first 20 entries

# # Save tensor if needed
# torch.save(best_target_tensor, "bestcase_target_classes_params-100_0_1000.pt")


# import numpy as np
# import matplotlib.pyplot as plt
# from stochastic_torch import *

# x = np.linspace(-1, 1, 100)           # normal numeric input
# bipolar_x = bip(x/4)                   # bipolar representation

# # --- Step 4: Apply activations ---
# tanh_normal = np.tanh(x)
# tanh_stochastic = bsn_actual_value(tanh_activation(bipolar_x))

# # --- Step 5: Plot results ---
# plt.figure(figsize=(8, 5))
# plt.plot(x, tanh_normal, label="Binary tanh", linewidth=2)
# plt.plot(x, tanh_stochastic, 'o', label="Stochastic tanh", alpha=0.4)
# plt.title("Binary vs Stochastic Tanh Activation")
# plt.xlabel("Input Value")
# plt.ylabel("Activation Output")
# plt.legend()
# plt.grid(True)
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from stochastic_torch import *

# x = np.linspace(-1, 1, 100)           # normal numeric input
# bipolar_x = bip(x/2)                   # bipolar representation

# # --- Step 4: Apply activations ---
# sigmoid_normal = (np.tanh(2*x)+1)/2
# sigmoid_stochastic = usn_actual_value(tanh_activation(bipolar_x))

# # --- Step 5: Plot results ---
# plt.figure(figsize=(8, 5))
# plt.plot(x, sigmoid_normal, label="Binary sigmoid", linewidth=2)
# plt.plot(x, sigmoid_stochastic, 'o', label="Stochastic sigmoid", alpha=0.4)
# plt.title("Binary vs Stochastic Sigmoid Activation")
# plt.xlabel("Input Value")
# plt.ylabel("Activation Output")
# plt.legend()
# plt.grid(True)
# plt.show()


# # extract boundary attack .pt files data
# import torch
# import pandas as pd

# # Load .pt file
# pt_file_path = 'boundary_adv_10samples_tensorattacks_batch10-10(0.00%_1000_1.5).pt'
# data = torch.load(pt_file_path, map_location='cpu')

# flat_data = {}

# for key, value in data.items():

#     # Case 1: tensor → convert to list
#     if torch.is_tensor(value):
#         flat_data[key] = value.tolist()

#     # Case 2: dictionary (only for "info")
#     elif isinstance(value, dict):
#         for subkey, subvalue in value.items():
#             flat_data[f"info_{subkey}"] = [subvalue]   # single-value column

#     # Case 3: list/other (rare)
#     else:
#         flat_data[key] = value

# # Find max column length for alignment
# max_len = max(len(v) if isinstance(v, list) else 1 for v in flat_data.values())

# # Normalize all columns to same length
# for key, value in flat_data.items():
#     if not isinstance(value, list):
#         flat_data[key] = [value] * max_len     # broadcast scalars
#     else:
#         if len(value) < max_len:
#             flat_data[key] = value + [None] * (max_len - len(value))

# # Create DataFrame
# df = pd.DataFrame(flat_data)

# # Save to Excel
# df.to_excel('sc_boundary_attack_batch10.xlsx', index=False)

# print("Saved clean Excel file with expanded tensors and dictionary values.")

# Merging boundary attack .pt files into 1

# code to merge all boundary atatck files
import torch
import os

# List of .pt files to merge
pt_files = [
    "boundary_adv_10samples_tensorattacks_batch1-1(0.00%_1000_1.5).pt",
    "boundary_adv_10samples_tensorattacks_batch2-2(10.00%_1000_1.5).pt",
    "boundary_adv_10samples_tensorattacks_batch3-3(0.00%_1000_1.5).pt",
    "boundary_adv_10samples_tensorattacks_batch4-4(0.00%_1000_1.5).pt",
    "boundary_adv_10samples_tensorattacks_batch5-5(0.00%_1000_1.5).pt",
    "boundary_adv_10samples_tensorattacks_batch6-6(0.00%_1000_1.5).pt",
    "boundary_adv_10samples_tensorattacks_batch7-7(10.00%_1000_1.5).pt",
    "boundary_adv_10samples_tensorattacks_batch8-8(10.00%_1000_1.5).pt",
    "boundary_adv_10samples_tensorattacks_batch9-9(0.00%_1000_1.5).pt",
    "boundary_adv_10samples_tensorattacks_batch10-10(0.00%_1000_1.5).pt",
    "boundary_adv_10samples_tensorattacks_batch11-11(0.00%_1000_1.5).pt",
    "boundary_adv_10samples_tensorattacks_batch12-12(0.00%_1000_1.5).pt",
]

merged = {}
initialized = False

for path in pt_files:
    print(f"Loading {path} ...")
    data = torch.load(path, map_location="cpu")

    for key, value in data.items():

        # Skip the 'info' dictionary
        if key == "info":
            continue

        # Ensure value is tensor
        if not torch.is_tensor(value):
            raise ValueError(f"Key '{key}' in {path} is not a tensor.")

        # Initialize first time
        if not initialized:
            merged[key] = [value]
        else:
            merged[key].append(value)

    initialized = True

# ---- Concatenate tensors for each key ----
for key in merged:
    merged[key] = torch.cat(merged[key], dim=0)

# Save merged result
output_path = "boundary_adv_120samples_tensorattacks_batch1-12(1000).pt"
torch.save(merged, output_path)

print("✓ Successfully merged.\nSaved as:", output_path)

