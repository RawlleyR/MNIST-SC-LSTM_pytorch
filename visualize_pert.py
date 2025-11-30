import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_adversarial_by_digit(pt_file_path, output_image_path='adversarial_visualization.png'):
    """
    Visualize the first occurrence of each digit (0-9) showing:
    - Original image with original label
    - Perturbation (difference)
    - Adversarial image with adversarial label
    
    Parameters:
    -----------
    pt_file_path : str
        Path to the .pt file containing the tensors
    output_image_path : str
        Path for the output image file (default: 'adversarial_visualization.png')
    """
    
    # Load the .pt file
    print(f"Loading data from {pt_file_path}...")
    data = torch.load(pt_file_path, map_location='cpu')
    
    # Extract tensors
    adv_images = data['adv_images']
    original_images = data['original_images']
    adv_labels = data['adv_labels']
    original_labels = data['original_labels']
    
    # Convert to numpy if tensors
    if torch.is_tensor(original_labels):
        original_labels = original_labels.cpu().numpy()
    if torch.is_tensor(adv_labels):
        adv_labels = adv_labels.cpu().numpy()
    
    # Find first occurrence of each digit (0-9) with significant perturbation
    digit_examples = {}
    perturbation_threshold = 1e-5
    
    for i in range(len(original_images)):
        orig_label = int(original_labels[i])
        
        # Only process if we haven't found this digit yet
        if orig_label not in digit_examples and orig_label >= 0 and orig_label <= 9:
            # Calculate L2 norm to check if perturbation is significant
            adv_flat = adv_images[i].flatten()
            orig_flat = original_images[i].flatten()
            l2_norm = torch.norm(adv_flat - orig_flat, p=2).item()
            
            # Only store if perturbation is >= threshold
            if l2_norm >= perturbation_threshold:
                digit_examples[orig_label] = {
                    'index': i,
                    'original_image': original_images[i],
                    'adv_image': adv_images[i],
                    'original_label': orig_label,
                    'adv_label': int(adv_labels[i]),
                    'l2_norm': l2_norm
                }
        
        # Break if we found all digits 0-9
        if len(digit_examples) == 10:
            break
    
    print(f"Found examples for {len(digit_examples)} digits")
    
    # Create visualization with 3 rows x 10 columns
    fig, axes = plt.subplots(3, 10, figsize=(20, 7))
    fig.suptitle('Adversarial Examples by Digit (0-9)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust spacing: minimal horizontal, larger vertical for text
    plt.subplots_adjust(wspace=0.05, hspace=0.55, left=0.02, right=0.98, top=0.94, bottom=0.05)
    
    for digit in range(10):
        if digit not in digit_examples:
            print(f"Warning: No example found for digit {digit}")
            # Hide axes for missing digits
            axes[0, digit].axis('off')
            axes[1, digit].axis('off')
            axes[2, digit].axis('off')
            continue
        
        example = digit_examples[digit]
        
        # Get images
        orig_img = example['original_image'].cpu().numpy()
        adv_img = example['adv_image'].cpu().numpy()
        
        # Handle different image formats (CHW or HW)
        if orig_img.ndim == 3:
            if orig_img.shape[0] == 1:  # Grayscale (1, H, W)
                orig_img = orig_img.squeeze(0)
                adv_img = adv_img.squeeze(0)
            elif orig_img.shape[0] == 3:  # RGB (3, H, W)
                orig_img = np.transpose(orig_img, (1, 2, 0))
                adv_img = np.transpose(adv_img, (1, 2, 0))
        
        # Calculate perturbation
        perturbation = adv_img - orig_img
        
        # Normalize perturbation for visualization
        # Make it visible by scaling
        pert_normalized = perturbation - perturbation.min()
        if pert_normalized.max() > 0:
            pert_normalized = pert_normalized / pert_normalized.max()
        
        # Determine if grayscale or color
        is_grayscale = orig_img.ndim == 2 or (orig_img.ndim == 3 and orig_img.shape[2] == 1)
        cmap = 'gray' if is_grayscale else None
        
        # Row 0: Original images
        axes[0, digit].imshow(orig_img, cmap=cmap)
        axes[0, digit].axis('off')
        # Display original label below image
        orig_text = f'Original\nLabel: {example["original_label"]}'
        axes[0, digit].text(0.5, -0.05, orig_text, 
                           transform=axes[0, digit].transAxes,
                           fontsize=14, ha='center', va='top',
                           fontweight='bold')
        
        # Row 1: Perturbation
        axes[1, digit].imshow(pert_normalized, cmap='hot')
        axes[1, digit].axis('off')
        # Display L2 norm below perturbation image
        l2_text = f'L2 Norm:\n{example["l2_norm"]:.4f}'
        axes[1, digit].text(0.5, -0.05, l2_text, 
                           transform=axes[1, digit].transAxes,
                           fontsize=14, ha='center', va='top',
                           fontweight='bold')
        
        # Row 2: Adversarial images
        axes[2, digit].imshow(adv_img, cmap=cmap)
        axes[2, digit].axis('off')
        # Display adversarial label below image
        adv_text = f'Adversarial\nLabel: {example["adv_label"]}'
        axes[2, digit].text(0.5, -0.05, adv_text, 
                           transform=axes[2, digit].transAxes,
                           fontsize=14, ha='center', va='top',
                           fontweight='bold')
    
    # Add row labels on the left side
    fig.text(0.01, 0.78, 'Original', fontsize=14, fontweight='bold', 
             rotation=90, va='center', ha='center')
    fig.text(0.01, 0.50, 'Perturbation', fontsize=14, fontweight='bold', 
             rotation=90, va='center', ha='center')
    fig.text(0.01, 0.22, 'Adversarial', fontsize=14, fontweight='bold', 
             rotation=90, va='center', ha='center')
    
    plt.savefig(output_image_path, dpi=200, bbox_inches='tight')
    print(f"\nVisualization saved to {output_image_path}")
    plt.show()
    
    # Print summary
    print("\nSummary of examples:")
    for digit in sorted(digit_examples.keys()):
        example = digit_examples[digit]
        print(f"Digit {digit}: Original label={example['original_label']}, "
              f"Adversarial label={example['adv_label']}, "
              f"L2 norm={example['l2_norm']:.6f}, "
              f"Index={example['index']}")

if __name__ == "__main__":
    pt_file_path = "bin_boundary_adv_200samples_tensorattacks_batch1-1(0.00%_5000_0.01_1.5).pt"
    output_path = "adversarial_examples_visualization(bin_boundary_5k).png"
    
    try:
        visualize_adversarial_by_digit(pt_file_path, output_path)
    except FileNotFoundError:
        print(f"Error: File '{pt_file_path}' not found. Please update the path.")
    except KeyError as e:
        print(f"Error: Expected key {e} not found in the .pt file.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")