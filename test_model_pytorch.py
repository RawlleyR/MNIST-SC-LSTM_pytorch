# test_model.py

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from CW_attack import Lstm_RNN 

def test_model(model_path, batch_size, seq_dim, input_dim, hidden_dim, hidden_dim2, output_dim, device):
    transform = transforms.Compose([transforms.ToTensor()])
    # testset = dsets.MNIST(root='./data', train=False, download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # load adversarial test dataset
    data = torch.load('cw_adversarial_100samples_tensorattacks(10%_10_1.6).pt')
    adv_images = data['adv_images']
    org_labels = data['original_labels']

    adv_dataset = torch.utils.data.TensorDataset(adv_images, org_labels)
    testloader = torch.utils.data.DataLoader(adv_dataset, batch_size=batch_size, shuffle=False)
    

    model = Lstm_RNN(batch_size, seq_dim, input_dim, hidden_dim, hidden_dim2, output_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images = images.view(-1, 28, 28).to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')
    
    return predicted, data['adv_labels'], org_labels

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print(device)
    predicted, adv_labels, org_labels = test_model(
        model_path='mnist_2_layer_adamax_train_gates_div_200-100hl.pth',
        batch_size=100,
        seq_dim=28,
        input_dim=28,
        hidden_dim=200,
        hidden_dim2=100,
        output_dim=10,
        device=device
    )
    
    print("Pred: ", predicted, "\n", "adv_: ", adv_labels, "\n", "org_: ", org_labels)
