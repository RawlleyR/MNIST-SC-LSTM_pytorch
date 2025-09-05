import torch
import pandas as pd

# Map target class → filename (update with your actual names)
file_paths = {
    0: "cw_targeted_0_adv_500samples_tensorattacks_batch1-5(39.6%_100_0_10000).pt",
    1: "cw_targeted_1_adv_500samples_tensorattacks_batch1-5(74.8%_100_0_10000).pt",
    2: "cw_targeted_2_adv_500samples_tensorattacks_batch1-5(78.4%_100_0_10000).pt",
    3: "cw_targeted_3_adv_500samples_tensorattacks_batch1-5(72.4%_100_0_10000).pt",
    4: "cw_targeted_4_adv_500samples_tensorattacks_batch1-5(55%_100_0_10000).pt",
    5: "cw_targeted_5_adv_500samples_tensorattacks_batch1-5(83%_100_0_10000).pt",
    6: "cw_targeted_6_adv_500samples_tensorattacks_batch1-5(37.4%_100_0_10000).pt",
    7: "cw_targeted_7_adv_500samples_tensorattacks_batch1-5(55%_100_0_10000).pt",
    8: "cw_targeted_8_adv_500samples_tensorattacks_batch1-5(90.6%_100_0_10000).pt",
    9: "cw_targeted_9_adv_500samples_tensorattacks_batch1-5(62.4%_100_0_10000).pt",
}

results = {}

for target_class, file_path in file_paths.items():
    data = torch.load(file_path)

    adv_labels = data['adv_labels']
    original_labels = data['original_labels']
    adv_images = data['adv_images']
    orig_images = data['original_images']

    for idx, (adv, orig, adv_img, orig_img) in enumerate(
        zip(adv_labels, original_labels, adv_images, orig_images)
    ):
        idx = int(idx)
        adv = int(adv)
        orig = int(orig)

        if idx not in results:
            results[idx] = {
                "Original Class": orig,
                "Successful Targets": [],
                "L2 Distances": []
            }

        if adv == target_class:  # successful attack
            # Compute L2 norm of perturbation
            perturb = adv_img - orig_img
            l2_dist = torch.norm(perturb.view(-1), p=2).item()

            results[idx]["Successful Targets"].append(target_class)
            results[idx]["L2 Distances"].append(l2_dist)

# Convert to DataFrame
rows = []
for idx, info in results.items():
    targets = info["Successful Targets"]
    distances = info["L2 Distances"]

    # Find target with min L2 (ignoring original class)
    best_target = None
    best_dist = float("inf")
    for t, d in zip(targets, distances):
        if t != info["Original Class"] and d < best_dist:
            best_target = t
            best_dist = d

    row = {
        "Image Index": idx,
        "Original Class": info["Original Class"],
        "Successful Targets": ", ".join(map(str, targets)),
        "L2 Distances": ", ".join(f"{d:.4f}" for d in distances),
        "Best Target (min L2)": best_target if best_target is not None else "None",
    }
    rows.append(row)

df = pd.DataFrame(rows)

# Save to Excel
df.to_excel("cw_attack_summary_with_best_target.xlsx", index=False)
print("✅ Excel file saved as cw_attack_summary_with_best_target.xlsx")
