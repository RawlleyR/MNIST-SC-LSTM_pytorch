import foolbox as fb
import torch
from CW_attack import main
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from tqdm import tqdm

from stoc_boundary_attack import SCHybridModel


def generate_boundary_adversarial_examples_gpu(model, testloader, device, 
                                               start_batch=0, num_batches=5,
                                               steps=5000, spherical_step=0.01, step_adaptation=1.5):
    """
    Runs Boundary Attack on a PyTorch model using Foolbox.

    Args:
        model: PyTorch model to attack.
        testloader: DataLoader for MNIST test set.
        device: "cuda" or "cpu".
        start_batch: batch index to start from.
        num_batches: how many batches to attack.
        steps: max iterations of Boundary Attack.
        spherical_step, step_adaptation: Foolbox hyperparameters.

    Returns:
        adv_images_all: tensor of adversarial images
        orig_images_all: tensor of original clean images
        predicted: tensor of model predictions on adversarial images
        labels_all: tensor of ground-truth labels
    """

    model.eval()
    model.to(device)

    # Foolbox wrapper
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))  # MNIST pixels are in [0,1]

    # Define the Boundary Attack
    atk = fb.attacks.BoundaryAttack(steps=steps,
                                    spherical_step=spherical_step,
                                    step_adaptation=step_adaptation)

    adv_images_all = []
    orig_images_all = []
    labels_all = []

    dataiter = iter(testloader)

    # Skip batches before start_batch
    for i in range(start_batch):
        try:
            next(dataiter)
        except StopIteration:
            print("Reached end of dataset while skipping batches.")
            return None

    for _ in tqdm(range(num_batches), desc=f"Running Boundary Attack (batches {start_batch + 1}-{start_batch + num_batches})"):
        try:
            images, labels = next(dataiter)
        except StopIteration:
            break  # End of dataset

        images, labels = images.to(device), labels.to(device)
        images = images.squeeze(1)

        # Foolbox attack: returns raw, clipped adversarial, and success mask
        raw_advs, clipped_advs, success = atk(fmodel, images, labels, epsilons=None)
        
        print(type(success))
        
        n_success = int(success.sum().item())      # if bool -> True counts as 1
        n_failure = success.numel() - n_success


        print(f"Successes: {n_success}, Failures: {n_failure}")

        adv_images_all.append(clipped_advs.detach().cpu())
        orig_images_all.append(images.detach().cpu())
        labels_all.append(labels.detach().cpu())

    # Concatenate all batches into single tensors
    adv_images_all = torch.cat(adv_images_all)
    orig_images_all = torch.cat(orig_images_all)
    labels_all = torch.cat(labels_all)

    # Evaluate attack accuracy
    with torch.no_grad():
        adv_images_all = adv_images_all.to(device)
        outputs = model(adv_images_all)
        predicted = torch.argmax(outputs, dim=1)
        acc = (predicted == labels_all.to(device)).float().mean().item()
        print(f"Accuracy on {num_batches * len(labels)} Boundary adversarial examples "
              f"(batches {start_batch + 1}–{start_batch + num_batches}): {acc * 100:.2f}%")

    return adv_images_all, orig_images_all, predicted, labels_all, acc



if __name__ == "__main__":
    
    batch_size = 100
    batch_size_test = 100
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
    start_batch = 0
    num_batches = 1
    
    steps=500
    spherical_step=0.01
    step_adaptation=1.5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # For SC model:
    # dictionary = torch.load('mnist_2_layer_adamax_train_cx_div2_200-100hl(98.43B&97.36SC).pth', map_location=device)

    # sc_model = SCHybridModel(dictionary, hidden_dim, hidden_dim2, output_dim, device=device)
    # sc_model.to(device).eval()

    # x_adv, x_test, y_adv, y_test, success = generate_boundary_adversarial_examples_gpu(
    # sc_model, testloader, device, start_batch, num_batches, steps, spherical_step, step_adaptation)

    # For Binary Model:
    model = main(True, trainloader, num_epochs, batch_size, seq_dim, input_dim, hidden_dim, hidden_dim2, output_dim, device)
    
    x_adv, x_test, y_adv, y_test, success = generate_boundary_adversarial_examples_gpu(
    model, testloader, device, start_batch, num_batches, steps, spherical_step, step_adaptation)
    
    # Check if channel dim exists
    if x_adv.ndim == 3:
        print("adding channel dimension to x_adv")
        x_adv = x_adv.unsqueeze(1)
        print(x_adv.size())
    if x_test.ndim == 3:
        print("adding channel dimension to x_test")
        x_test = x_test.unsqueeze(1)
        print(x_test.size())
        
    torch.save({
        'adv_images': x_adv.clone().detach().cpu(),  # in case it's NumPy
        'original_images': x_test.clone().detach().cpu(),
        'adv_labels': y_adv.clone().detach().cpu(),
        'original_labels': y_test.clone().detach().cpu()
    }, f'boundary_adv_{num_batches*batch_size_test}samples_tensorattacks_batch{start_batch+1}-{start_batch+num_batches}({success*100:.2f}%).pt')

    print(f"Adversarial samples saved to boundary_samples.pt")