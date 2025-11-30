import torch
import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment

def calculate_l2_norms(pt_file_path, output_excel_path='l2_norms.xlsx'):
    """
    Calculate L2 norms between adversarial and original images from a .pt file
    and save results to an Excel file.
    
    Parameters:
    -----------
    pt_file_path : str
        Path to the .pt file containing the tensors
    output_excel_path : str
        Path for the output Excel file (default: 'l2_norms.xlsx')
    """
    
    # Load the .pt file
    print(f"Loading data from {pt_file_path}...")
    data = torch.load(pt_file_path, map_location='cpu')
    
    # Extract tensors
    adv_images = data['adv_images']
    original_images = data['original_images']
    adv_labels = data['adv_labels']
    original_labels = data['original_labels']
    
    print(f"Number of image pairs: {len(adv_images)}")
    
    # Calculate L2 norms for each image pair
    l2_norms = []
    
    for i in range(len(adv_images)):
        # Flatten images and calculate L2 norm (Euclidean distance)
        adv_flat = adv_images[i].flatten()
        orig_flat = original_images[i].flatten()
        
        # Calculate L2 norm: sqrt(sum((adv - orig)^2))
        l2_norm = torch.norm(adv_flat - orig_flat, p=2).item()
        l2_norms.append(l2_norm)
    
    # Filter out zero values for average calculation
    non_zero_norms = [norm for norm in l2_norms if norm > 1e-05]
    
    if non_zero_norms:
        avg_l2_norm = np.mean(non_zero_norms)
    else:
        avg_l2_norm = 0
    
    print(f"Average L2 norm (excluding zeros): {avg_l2_norm:.6f}")
    print(f"Total perturbations: {len(l2_norms)}")
    print(f"Non-zero perturbations: {len(non_zero_norms)}")
    
    # Create DataFrame
    df = pd.DataFrame({
        'Image Index': range(len(l2_norms)),
        'Original Label': original_labels.cpu().numpy() if torch.is_tensor(original_labels) else original_labels,
        'Adversarial Label': adv_labels.cpu().numpy() if torch.is_tensor(adv_labels) else adv_labels,
        'L2 Norm (Perturbation Size)': l2_norms
    })
    
    # Create Excel file with formatting
    print(f"Creating Excel file at {output_excel_path}...")
    
    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='L2 Norms', index=False, startrow=3)
        
        workbook = writer.book
        worksheet = writer.sheets['L2 Norms']
        
        # Add title and summary statistics
        worksheet['A1'] = 'L2 Norm Analysis: Adversarial vs Original Images'
        worksheet['A1'].font = Font(size=14, bold=True)
        
        worksheet['A2'] = f'Average L2 Norm (excluding zeros): {avg_l2_norm:.6f}'
        worksheet['A2'].font = Font(size=11, bold=True)
        worksheet['B2'] = f'Total Images: {len(l2_norms)}'
        worksheet['C2'] = f'Non-zero Perturbations: {len(non_zero_norms)}'
        
        # Format header row
        header_fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
        header_font = Font(color='FFFFFF', bold=True)
        
        for cell in worksheet[4]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal='center')
        
        # Adjust column widths
        worksheet.column_dimensions['A'].width = 15
        worksheet.column_dimensions['B'].width = 18
        worksheet.column_dimensions['C'].width = 20
        worksheet.column_dimensions['D'].width = 28
        
        # Format L2 norm values to 6 decimal places
        for row in range(5, len(l2_norms) + 5):
            worksheet[f'D{row}'].number_format = '0.000000'
    
    print(f"Excel file saved successfully at {output_excel_path}")
    print("\nSummary:")
    print(f"  - Total image pairs processed: {len(l2_norms)}")
    print(f"  - Non-zero perturbations: {len(non_zero_norms)}")
    print(f"  - Zero perturbations: {len(l2_norms) - len(non_zero_norms)}")
    print(f"  - Average L2 norm (excluding zeros): {avg_l2_norm:.6f}")
    
    return df, avg_l2_norm

# Example usage:
if __name__ == "__main__":
    pt_file_path = "bin_boundary_adv_100samples_tensorattacks_batch1-1(0.00%_1000_1.5).pt"
    output_path = "l2_norms_results_bin_boundary 1000.xlsx"
    
    try:
        df, avg_norm = calculate_l2_norms(pt_file_path, output_path)
        print("\nFirst few rows of the data:")
        print(df.head())
    except FileNotFoundError:
        print(f"Error: File '{pt_file_path}' not found. Please update the path.")
    except KeyError as e:
        print(f"Error: Expected key {e} not found in the .pt file.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")