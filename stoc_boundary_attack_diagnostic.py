import torch
import torch.nn as nn
import foolbox as fb
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from stoc_boundary_attack import SCHybridModel
import torchvision.datasets as dsets
import torchvision.transforms as transforms

def diagnose_boundary_attack_behavior(model, testloader, device,
                                      start_batch=0, num_batches=1,
                                      steps=5000, spherical_step=0.01,
                                      step_adaptation=1.5,
                                      num_stochastic_tests=10):
    """
    Diagnostic version that tracks:
    1. Whether initial adversarial is found
    2. Perturbation size progression
    3. Classification consistency (for stochastic models)
    4. Why attacks fail
    """
    model.eval()
    model.to(device)

    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    
    diagnostics = []

    dataiter = iter(testloader)
    for _ in range(start_batch):
        try:
            next(dataiter)
        except StopIteration:
            return None

    for batch_idx in tqdm(range(num_batches), desc="Diagnostic batches"):
        try:
            images, labels = next(dataiter)
        except StopIteration:
            break

        if images.dim() == 4 and images.size(1) == 1:
            images = images.squeeze(1)

        batch_size = images.size(0)
        for i in tqdm(range(batch_size), desc=f"Diagnosing images", leave=False):
            x_orig = images[i].unsqueeze(0).to(device)
            y_true = labels[i].item()
            
            diag = {
                'image_idx': i,
                'true_label': y_true,
                'initial_adversarial_found': False,
                'attack_success': False,
                'initial_perturbation': None,
                'final_perturbation': None,
                'perturbation_reduction': None,
                'stochastic_consistency': {},
                'failure_reason': 'unknown'
            }
            
            # Test original image classification consistency
            orig_preds = []
            with torch.no_grad():
                for _ in range(num_stochastic_tests):
                    logits = model(x_orig)
                    pred = logits.argmax(dim=1).item()
                    orig_preds.append(pred)
            
            diag['original_predictions'] = orig_preds
            diag['original_pred_mode'] = max(set(orig_preds), key=orig_preds.count)
            diag['original_consistency'] = orig_preds.count(diag['original_pred_mode']) / len(orig_preds)
            
            print(f"\nImage {i}: True label={y_true}, Predictions={orig_preds}")
            print(f"  Consistency: {diag['original_consistency']*100:.1f}%")
            
            # Try to find initial adversarial manually
            initial_adv = None
            for attempt in range(100):
                random_img = torch.rand_like(x_orig)
                with torch.no_grad():
                    pred = model(random_img).argmax(dim=1).item()
                if pred != y_true:
                    initial_adv = random_img
                    initial_pert = torch.norm(initial_adv - x_orig, p=2).item()
                    diag['initial_adversarial_found'] = True
                    diag['initial_perturbation'] = initial_pert
                    print(f"  Found initial adversarial: pred={pred}, pert={initial_pert:.4f}")
                    break
            
            if not diag['initial_adversarial_found']:
                diag['failure_reason'] = 'no_initial_adversarial'
                print(f"  FAILED: Could not find initial adversarial")
                diagnostics.append(diag)
                continue
            
            # Test initial adversarial consistency
            init_adv_preds = []
            with torch.no_grad():
                for _ in range(num_stochastic_tests):
                    logits = model(initial_adv)
                    pred = logits.argmax(dim=1).item()
                    init_adv_preds.append(pred)
            
            diag['initial_adv_predictions'] = init_adv_preds
            diag['initial_adv_consistency'] = init_adv_preds.count(init_adv_preds[0]) / len(init_adv_preds)
            print(f"  Initial adv predictions: {init_adv_preds}")
            print(f"  Initial adv consistency: {diag['initial_adv_consistency']*100:.1f}%")
            
            # Run Foolbox boundary attack
            atk = fb.attacks.BoundaryAttack(
                steps=steps,
                spherical_step=spherical_step,
                step_adaptation=step_adaptation
            )
            
            try:
                raw_advs, clipped_advs, success = atk(fmodel, x_orig, 
                                                      torch.tensor([y_true]).to(device), 
                                                      epsilons=None)
                
                final_adv = clipped_advs[0]
                final_pert = torch.norm(final_adv - x_orig[0], p=2).item()
                diag['final_perturbation'] = final_pert
                diag['attack_success'] = bool(success[0])
                
                if diag['initial_perturbation'] is not None:
                    diag['perturbation_reduction'] = diag['initial_perturbation'] - final_pert
                    diag['perturbation_reduction_percent'] = (diag['perturbation_reduction'] / diag['initial_perturbation']) * 100
                
                print(f"  Attack success: {diag['attack_success']}")
                print(f"  Final perturbation: {final_pert:.4f}")
                if diag['perturbation_reduction'] is not None:
                    print(f"  Reduction: {diag['perturbation_reduction']:.4f} ({diag['perturbation_reduction_percent']:.1f}%)")
                
                # Test final adversarial consistency
                final_preds = []
                with torch.no_grad():
                    for _ in range(num_stochastic_tests):
                        logits = model(final_adv.unsqueeze(0))
                        pred = logits.argmax(dim=1).item()
                        final_preds.append(pred)
                
                diag['final_adv_predictions'] = final_preds
                diag['final_adv_mode'] = max(set(final_preds), key=final_preds.count)
                diag['final_adv_consistency'] = final_preds.count(diag['final_adv_mode']) / len(final_preds)
                
                print(f"  Final adv predictions: {final_preds}")
                print(f"  Final adv consistency: {diag['final_adv_consistency']*100:.1f}%")
                
                # Determine failure reason
                if not diag['attack_success']:
                    if diag['final_adv_mode'] == y_true:
                        if diag['final_adv_consistency'] < 0.8:
                            diag['failure_reason'] = 'stochastic_inconsistency'
                            print(f"  REASON: Stochastic model gives inconsistent predictions")
                        else:
                            diag['failure_reason'] = 'crossed_back_to_correct'
                            print(f"  REASON: Attack moved back to correct class region")
                    elif diag['perturbation_reduction'] is not None and diag['perturbation_reduction'] < 0.01:
                        diag['failure_reason'] = 'no_progress'
                        print(f"  REASON: Attack made no progress in reducing perturbation")
                    else:
                        diag['failure_reason'] = 'other'
                        print(f"  REASON: Unknown - check manually")
                
            except ValueError as e:
                diag['failure_reason'] = 'foolbox_error'
                print(f"  ERROR: Foolbox raised ValueError: {e}")
            
            diagnostics.append(diag)
    
    return diagnostics


def analyze_diagnostics(diagnostics):
    """Analyze and summarize diagnostic results"""
    total = len(diagnostics)
    
    init_found = sum(1 for d in diagnostics if d['initial_adversarial_found'])
    attack_success = sum(1 for d in diagnostics if d['attack_success'])
    
    # Categorize failures
    failure_reasons = {}
    for d in diagnostics:
        if not d['attack_success']:
            reason = d['failure_reason']
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    
    # Calculate average perturbation reductions
    reductions = [d['perturbation_reduction'] for d in diagnostics 
                  if d['perturbation_reduction'] is not None]
    
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    print(f"Total images tested: {total}")
    print(f"Initial adversarial found: {init_found}/{total} ({init_found/total*100:.1f}%)")
    print(f"Attack success: {attack_success}/{total} ({attack_success/total*100:.1f}%)")
    print(f"\nFailure reasons:")
    for reason, count in failure_reasons.items():
        print(f"  {reason}: {count} ({count/total*100:.1f}%)")
    
    if reductions:
        print(f"\nPerturbation reduction statistics:")
        print(f"  Mean: {np.mean(reductions):.4f}")
        print(f"  Median: {np.median(reductions):.4f}")
        print(f"  Min: {np.min(reductions):.4f}")
        print(f"  Max: {np.max(reductions):.4f}")
    
    # Check stochastic consistency
    orig_consistencies = [d['original_consistency'] for d in diagnostics]
    print(f"\nOriginal image prediction consistency:")
    print(f"  Mean: {np.mean(orig_consistencies)*100:.1f}%")
    
    return failure_reasons


def plot_diagnostics(diagnostics, save_path='diagnostic_plots.png'):
    """Create visualization of diagnostic results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Perturbation progression
    ax = axes[0, 0]
    images_with_progress = [d for d in diagnostics 
                           if d['initial_perturbation'] is not None 
                           and d['final_perturbation'] is not None]
    
    if images_with_progress:
        indices = range(len(images_with_progress))
        initial_perts = [d['initial_perturbation'] for d in images_with_progress]
        final_perts = [d['final_perturbation'] for d in images_with_progress]
        
        ax.plot(indices, initial_perts, 'o-', label='Initial', color='red', alpha=0.7)
        ax.plot(indices, final_perts, 's-', label='Final', color='blue', alpha=0.7)
        ax.set_xlabel('Image Index')
        ax.set_ylabel('L2 Perturbation')
        ax.set_title('Perturbation Size: Initial vs Final')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Perturbation reduction
    ax = axes[0, 1]
    reductions = [d['perturbation_reduction_percent'] for d in diagnostics 
                  if d.get('perturbation_reduction_percent') is not None]
    if reductions:
        ax.hist(reductions, bins=20, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Perturbation Reduction (%)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Perturbation Reductions')
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Prediction consistency
    ax = axes[1, 0]
    orig_cons = [d['original_consistency'] for d in diagnostics]
    if any('final_adv_consistency' in d for d in diagnostics):
        final_cons = [d['final_adv_consistency'] for d in diagnostics 
                     if 'final_adv_consistency' in d]
        ax.scatter(orig_cons[:len(final_cons)], final_cons, alpha=0.6)
        ax.set_xlabel('Original Prediction Consistency')
        ax.set_ylabel('Final Adversarial Consistency')
        ax.set_title('Prediction Consistency Comparison')
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Failure reasons pie chart
    ax = axes[1, 1]
    failure_reasons = {}
    for d in diagnostics:
        if not d['attack_success']:
            reason = d['failure_reason']
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    
    if failure_reasons:
        ax.pie(failure_reasons.values(), labels=failure_reasons.keys(), 
               autopct='%1.1f%%', startangle=90)
        ax.set_title('Failure Reasons Distribution')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nDiagnostic plots saved to {save_path}")


# Example usage
if __name__ == "__main__":
    batch_size = 100
    batch_size_test = 10
    # list all transformations
    transform = transforms.Compose([transforms.ToTensor()])

    # download and load training dataset
    trainset = dsets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    # download and load testing dataset
    testset = dsets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=2)

    input_dim = 28
    hidden_dim = 200
    hidden_dim2 = 100
    output_dim = 10
    seq_dim = 28
    num_epochs = 20
    start_batch = 1
    num_batches = 1
    
    steps=1000
    spherical_step=0.01
    step_adaptation=1.5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # For SC model:
    dictionary = torch.load('mnist_2_layer_adamax_train_cx_div2_200-100hl(98.43B&97.36SC).pth', map_location=device)

    sc_model = SCHybridModel(dictionary, hidden_dim, hidden_dim2, output_dim, device=device)
    sc_model.to(device).eval()
    
    # Run diagnostic
    diagnostics = diagnose_boundary_attack_behavior(
        model=sc_model,
        testloader=testloader,
        device=device,
        start_batch=start_batch,
        num_batches=num_batches,
        steps=steps,
        spherical_step=spherical_step,
        step_adaptation=step_adaptation,
        num_stochastic_tests=10  # Test each image 10 times for consistency
    )
    
    # Analyze results
    failure_reasons = analyze_diagnostics(diagnostics)
    
    # Plot results
    plot_diagnostics(diagnostics, save_path='boundary_diagnostics.png')
    
    # Save detailed results
    torch.save(diagnostics, 'boundary_diagnostics_detailed.pt')