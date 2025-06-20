import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.optim as optim
import math

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import CarliniL2Method
import torch.nn.functional as F
import torchattacks
from tqdm import tqdm  # Optional: for a progress bar

class CustomLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.W = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias_w = nn.Parameter(torch.Tensor(hidden_sz * 4))
        self.bias_u = nn.Parameter(torch.Tensor(hidden_sz * 4))
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x,
                init_states=None):
        """Assumes x is of shape (sequence, batch, feature)"""
        seq_sz, bs, _ = x.size()

        if init_states is None:
            h_t, c_t = (torch.zeros(1, bs, self.hidden_size).to(x.device),
                        torch.zeros(1, bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
        # print('h_t,c_t',h_t.shape)
        HS = self.hidden_size

        hidden_seq = []

        h_t = h_t[0, :, :]
        c_t = c_t[0, :, :]

        for t in range(seq_sz):
            x_t = x[t, :, :]

            maximum = 1.0

            '''
            p_x = int(self.W.t().shape[0] / Lb)
            q_x = int(self.W.t().shape[1] / Lb)
            m_x = int(x_t.shape[0])
            p_h = int(self.U.t().shape[0] / Lb)
            q_h = int(self.U.t().shape[1] / Lb)
            m_h = int(h_t.shape[0])
            '''

            gates = x_t @ (self.W) + h_t @ (self.U) + self.bias_w + self.bias_u

            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]*2),  # input
                torch.sigmoid(gates[:, HS:HS * 2]*2),  # forget
                torch.tanh(gates[:, HS * 2:HS * 3]*2),
                torch.sigmoid(gates[:, HS * 3:]*2),  # output
            )
            c_t = (f_t * c_t + i_t * g_t)
            h_t = o_t * torch.tanh(c_t)
            for t in c_t:
                maximum = max(torch.max(torch.abs(t)), maximum)
            #  print('max',maximum)

            c_t = torch.div(c_t, maximum)
            # c_t = torch.div(c_t, 2)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        return hidden_seq, (h_t, c_t)


class Lstm_RNN(nn.Module):
    def __init__(self, batch_size, n_steps, n_inputs, n_neurons, n_neurons2, n_outputs):
        super(Lstm_RNN, self).__init__()

        self.n_neurons = n_neurons
        self.n_neurons2 = n_neurons2
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        # self.layer_dim = layer_dim
        self.lstm1 = CustomLSTM(self.n_inputs, self.n_neurons)
        self.lstm2 = CustomLSTM(self.n_neurons, self.n_neurons2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        # self.basic_rnn = nn.RNN(self.n_inputs, self.n_neurons)

        self.FC = nn.Linear(self.n_neurons2, self.n_outputs)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def init_hidden(self, n_neurons):
        # (num_layers, batch_size, n_neurons)
        device = next(self.parameters()).device
        return (torch.zeros(1,self.batch_size, n_neurons).to(device),
                torch.zeros(1,self.batch_size, n_neurons).to(device))

    def forward(self, X):
        # transforms X to dimensions: n_steps X batch_size X n_inputs
        X = X.permute(1, 0, 2)
        #print('X', X.shape)
        self.batch_size = X.size(1)
        self.hidden1, self.cell_state1 = self.init_hidden(self.n_neurons)
        self.hidden2, self.cell_state2 = self.init_hidden(self.n_neurons2)

        lstm_out1, (self.hidden1, self.cell_state1) = self.lstm1(X, (self.hidden1, self.cell_state1))
        drop_out1 = self.dropout1(lstm_out1)
        # print('lstm', drop_out1.shape)
        lstm_out2, (self.hidden2, self.cell_state2) = self.lstm2(drop_out1, (self.hidden2, self.cell_state2))
        drop_out2=self.dropout2(lstm_out2)
        # print('lstm2', drop_out2.shape)
        out = self.FC(drop_out2[-1, :, :])
        # Softmax funcrtion isn't explicitly used because it is implicitly taken care of by the nn.CrossEntropyLoss()
        # print('lstm', out.view(-1, self.n_outputs).shape)
        return out.view(-1, self.n_outputs)  # batch_size X n_output


def get_accuracy(logit, target, batch_size):
    ''' Obtain accuracy for training round '''
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects / batch_size
    return accuracy.item()


def main(pretrained,trainloader,epochs, batch_size, seq_dim, input_dim, hidden_dim,hidden_dim2, output_dim, device):
    '''
        :param pretrained: if True: impelements already present trained model
        :param train_inout_seq: input data
        :param epochs: no of epochs
        :param model: model
        :param loss_function:
        :param optimizer:
        :return: trained model
    '''

    model = Lstm_RNN(batch_size, seq_dim, input_dim, hidden_dim,hidden_dim2, output_dim)

    device = device
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adamax(model.parameters(), lr=0.01)

    # Training

    if pretrained == False:
        for epoch in range(epochs):  # loop over the dataset multiple times
            train_running_loss = 0.0
            train_acc = 0.0
            model.train()

            # TRAINING ROUND
            for i, data in enumerate(trainloader):
                # zero the parameter gradients
                optimizer.zero_grad()

                # # reset hidden states
                # model.hidden = model.init_hidden(hidden_dim)

                # get the inputs
                inputs, labels = data
                inputs = inputs.view(-1, 28, 28).to(device)
                labels = labels.to(device)

                # forward + backward + optimize
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_running_loss += loss.detach().item()
                train_acc += get_accuracy(outputs, labels, batch_size)

            model.eval()
            print('Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f'
                  % (epoch, train_running_loss / i, train_acc / i))

        torch.save(model.state_dict(), 'mnist_2_layer_adamax_train_cx_div2_200-100hl.pth')  ### Saving  the model
    else:
        ### If pretrained=True, Load and use the trained model to predict the values
        model.load_state_dict(torch.load('mnist_2_layer_adamax_train_gates_div_200-100hl.pth', map_location=device))
        model = model.to(device)

    return model

def generate_cw_adversarial_examples(model, testloader, device):
    model.eval()
    model.to(device)

    # Define loss and optimizer for ART
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adamax(model.parameters(), lr=0.01)

    classifier = PyTorchClassifier(
        model=model,
        loss=loss_fn,
        optimizer=optimizer,
        input_shape=(28, 28),  # For MNIST
        nb_classes=10,
        clip_values=(0.0, 1.0),
    )

    # Get a batch of test data
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images = images.to(device)
    labels = labels.to(device)

    # Convert to NumPy
    x_test = images.squeeze(1).cpu().numpy()
    y_test = F.one_hot(labels, num_classes=10).cpu().numpy()

    # Create the attack
    attack = CarliniL2Method(
        classifier=classifier,
        targeted=False,
        max_iter=1000,  # Increase for stronger attacks
        binary_search_steps=10,
        learning_rate=0.01,
        confidence=0.5
    )

    # Generate adversarial samples
    x_adv = attack.generate(x=x_test)

    # Evaluate adversarial accuracy
    preds_adv = classifier.predict(x_adv)
    acc = np.mean(np.argmax(preds_adv, axis=1) == np.argmax(y_test, axis=1))
    print(f"Accuracy on C&W adversarial examples: {acc * 100:.2f}%")

    return x_adv, x_test, y_test

# def generate_cw_adversarial_examples_gpu(model, testloader, device):
#     model.eval()
#     model.to(device)

#     dataiter = iter(testloader)
#     images, labels = next(dataiter)
#     images = images.to(device)
#     labels = labels.to(device)
#     # print(images.size())  # (100, 1, 28, 28)

#     # # ✅ Confirm on GPU
#     # print("Model device:", next(model.parameters()).device)
#     # print("Images device:", images.device)
#     # print("Labels device:", labels.device)

#     # Create GPU-accelerated C&W attack
#     atk_images = images.squeeze(1)
#     atk = torchattacks.CW(model, c=10, kappa=1, steps=1000, lr=0.01)
#     # atk.set_return_type('float')  # Default is 'int', which disables gradients

#     # Generate adversarial samples
#     adv_images = atk(atk_images, labels)

#     # Evaluate accuracy
#     with torch.no_grad():
#         outputs = model(adv_images)
#         predicted = torch.argmax(outputs, dim=1)
#         acc = (predicted == labels).float().mean().item()
#         print(f"Accuracy on GPU C&W adversarial examples: {acc * 100:.2f}%")

#     return adv_images, images, labels


def generate_cw_adversarial_examples_gpu(model, testloader, device, num_batches=5):
    model.eval()
    model.to(device)

    # Define the CW attack
    atk = torchattacks.CW(model, c=10, kappa=1, steps=1000, lr=0.01)

    adv_images_all = []
    orig_images_all = []
    labels_all = []

    dataiter = iter(testloader)

    for _ in tqdm(range(num_batches), desc="Running CW Attack"):
        try:
            images, labels = next(dataiter)
        except StopIteration:
            break  # End of dataset

        images = images.to(device)
        labels = labels.to(device)
        images = images.squeeze(1)  # For MNIST: (N, 1, 28, 28) → (N, 28, 28)

        # Generate adversarial examples
        adv_images = atk(images, labels)

        adv_images_all.append(adv_images.detach().cpu())
        orig_images_all.append(images.detach().cpu())
        labels_all.append(labels.detach().cpu())

    # Concatenate all batches into single tensors
    adv_images_all = torch.cat(adv_images_all)
    orig_images_all = torch.cat(orig_images_all)
    labels_all = torch.cat(labels_all)

    # Evaluate attack accuracy (optional)
    with torch.no_grad():
        adv_images_all = adv_images_all.to(device)
        outputs = model(adv_images_all)
        predicted = torch.argmax(outputs, dim=1)
        acc = (predicted == labels_all.to(device)).float().mean().item()
        print(f"Accuracy on {num_batches * len(labels)} CW adversarial examples: {acc * 100:.2f}%")

    return adv_images_all, orig_images_all, predicted, labels_all


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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = main(True, trainloader, num_epochs, batch_size, seq_dim, input_dim, hidden_dim, hidden_dim2, output_dim, device)
    
    # === Generate adversarial samples ===
    # x_adv, x_test, y_test = generate_cw_adversarial_examples(model, testloader, device)
    
    x_adv, x_test, y_adv, y_test = generate_cw_adversarial_examples_gpu(model, testloader, device)
    
    # Check if channel dim exists
    if x_adv.ndim == 3:
        print("adding channel dimension to x_adv")
        x_adv = x_adv.squeeze(1)
        print(x_adv.size())
    if x_test.ndim == 3:
        print("adding channel dimension to x_test")
        x_test = x_test.squeeze(1)
        print(x_test.size())
    
    # Save them
    # torch.save({
    #     'adv_images': torch.tensor(x_adv).detach().cpu(),  # in case it's NumPy
    #     'original_images': torch.tensor(x_test).detach().cpu(),
    #     'original_labels': torch.tensor(y_test).detach().cpu()
    # }, 'cwl2_adversarial_samples_ART.pt')
    
    torch.save({
        'adv_images': x_adv.clone().detach().cpu(),  # in case it's NumPy
        'original_images': x_test.clone().detach().cpu(),
        'adv_labels': y_adv.clone().detach().cpu(),
        'original_labels': y_test.clone().detach().cpu()
    }, 'cw_adversarial_samples_tensorattacks.pt')

    print("Adversarial samples saved to cw_adversarial_samples.pt")