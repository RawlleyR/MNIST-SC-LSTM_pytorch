import torch
import matplotlib.pyplot as plt
import numpy as np

# Load file (change path as needed)
cw_attack = torch.load("cw_adversarial_500samples_tensorattacks(14.6%_10_1).pt")  # Or np.load("adv_images.npy")

# If it's torch tensor
adv_images = cw_attack['adv_images'].squeeze(1).detach().cpu()  # Shape: [N, 28, 28]
test_images = cw_attack['original_images'].squeeze(1).detach().cpu()
adv_preds = cw_attack['adv_labels'].detach().cpu()
org_labels = cw_attack['original_labels'].detach().cpu()

# Plot
plt.figure(figsize=(15, 4))
for i in range(10):
    plt.subplot(2, 10, i + 1)
    plt.imshow(test_images[i], cmap='gray')
    plt.axis('off')
    plt.title(f"Label: {org_labels[i].item()}", fontsize=8)
    # if i == 4:
    #     plt.title("Original\nLabel")
    plt.subplot(2, 10, i + 11)
    plt.imshow(adv_images[i], cmap='gray')
    plt.axis('off')
    plt.title(f"Pred: {adv_preds[i].item()}", fontsize=8)
    # if i == 4:
    #     plt.title("Adversarial\nPred")
    plt.axis('off')
plt.suptitle("First 10 Adversarial Images")
plt.show()
