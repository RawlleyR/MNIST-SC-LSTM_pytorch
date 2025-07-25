## here the Optimizers is changed from ADAM to ADAMAX
## stochastic only done for the activation fun


import torch
import torch.nn as nn

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from fft_ver4 import *
# %matplotlib inline
from stochastic import *
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
from Model import LSTM
import math
from apc import bsn_apc_2in_sum

'''
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.init_linear = nn.Linear(input_dim, input_dim)
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, layer_dim, batch_first=True)
        self.gru=nn.GRU(input_dim, hidden_dim, 1, batch_first=True)
        self.drop=nn.Dropout(0.2)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        # Initialize hidden state with zeros

        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        h1 = torch.zeros(1, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c1 = torch.zeros(1, x.size(0), self.hidden_dim).requires_grad_()

        linear_input = self.init_linear(x)
        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        print(x.size())
        out1, (hn, cn) = self.lstm1(linear_input, (h0.detach(), c0.detach()))
        #print(out1.size(),hn.size(),cn.size())
        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        #out2, hn1 = self.gru(x, h0.detach())
        #drop_out = self.drop(out1)
        #print(out.size(), hn1.size())
        #out2, (hn1, cn1) = self.lstm2(drop_out, (h1.detach(), c1.detach()))
        #print(out1[:, -1, :])
        out = self.fc(out1[:, -1, :])

        #print(out.size()) ##--> 100, 10
        return out

'''


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

            gates = x_t @ (self.W) + h_t @ (self.U) + self.bias_w + self.bias_u

            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]*2),  # input
                torch.sigmoid(gates[:, HS:HS * 2]*2),  # forget
                torch.tanh(gates[:, HS * 2:HS * 3]*2),
                torch.sigmoid(gates[:, HS * 3:]*2),  # output
            )
            c_t = (f_t * c_t + i_t * g_t)
            h_t = o_t * torch.tanh(c_t)
            
            # With sample-wise normalization:
            # (batch dependant normalization results in inconsistencies during testing with different batch sizes)
            # maximum = torch.max(torch.abs(c_t), dim=-1, keepdim=True)[0]
            # maximum = torch.clamp(maximum, min=1.0)
            # #  print('max',maximum)

            # c_t = torch.div(c_t, maximum)
            c_t = torch.div(c_t, 2)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        
        return hidden_seq, (h_t, c_t)


class Lstm_RNN(nn.Module):
    def __init__(self, batch_size, n_steps, n_inputs, n_neurons, n_neurons2, n_outputs, layer_dim):
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

        self.FC = nn.Linear(self.n_neurons2, self.n_outputs)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def init_hidden(self, n_neurons):
        # (num_layers, batch_size, n_neurons)
        return (torch.zeros(1,self.batch_size, n_neurons),
                torch.zeros(1,self.batch_size, n_neurons))

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


def main(pretrained,trainloader,epochs, batch_size, seq_dim, input_dim, hidden_dim,hidden_dim2, output_dim, layer_dim):
    '''
        :param pretrained: if True: impelements already present trained model
        :param train_inout_seq: input data
        :param epochs: no of epochs
        :param model: model
        :param loss_function:
        :param optimizer:
        :return: trained model
    '''

    model = Lstm_RNN(batch_size, seq_dim, input_dim, hidden_dim,hidden_dim2, output_dim, layer_dim)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

                # reset hidden states
                model.hidden = model.init_hidden(hidden_dim)

                # get the inputs
                inputs, labels = data
                inputs = inputs.view(-1, 28, 28)
                labels = labels

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
        model.load_state_dict(torch.load('mnist_2_layer_adamax_train_cx_div2_200-100hl(98.43B&97.36SC).pth'))

    return model



def Linear(x, weight, bias):
    '''
        :param x: first value to be multiplied
        :param weight: second value to be multiplied
        :param bias: value to add
        :return: actual value of a stochastic number so obtained
    '''

    out = x @ weight.t() + bias
    return out


def lstm_stoc_activation(input1,input2, lstm_size, hx, cx,hx_n,cx_n, weight_ih, weight_hh, bias_ih, bias_hh):
    '''
    :param input: input
    :param lstm_size: hidden layer lize
    :param hx: hidden cell state 1
    :param cx: hidden cell state 2
    :param weight_ih: input weight
    :param weight_hh: hidden layer weight
    :param bias_ih: input bias
    :param bias_hh: hidden layer bias
    :return: hx

    '''

    # type: (Tensor, Tuple[Tensor, Tensor], Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]

    seq_sz, bs, _ = input1.size()
    hidden_seq = []
    hidden_seq_n = []
    hx = hx[0, :, :]
    cx = cx[0, :, :]
    hx_n = hx_n[0, :, :]
    cx_n = cx_n[0, :, :]
    HS = lstm_size
    for t in range(seq_sz):
        x_t = input1[t, :, :]
        x_t_n = input2[t, :, :]


        #print('shape', x_t.shape)

        maximum=1.0
        maximum_n=1.0
        # print('shape', weight_ih.t().shape)
        # batch the computations into a single matrix multiplication
        gates = x_t @ weight_ih + hx @ weight_hh + bias_ih + bias_hh
        gates_normal=x_t_n @ weight_ih + hx_n @ weight_hh + bias_ih + bias_hh

        # print('gates_it', gates[0][0:4], gates_normal[0][0:4])
        # print('gates_ft', gates[0][HS:HS+4], gates_normal[0][HS:HS+4])
        # print('gates_gt', gates[0][HS * 2:HS * 2+4], gates_normal[0][HS * 2:HS * 2+4])
        # print('gates_ot', gates[0][HS * 3:HS * 3+4], gates_normal[0][HS * 3:HS * 3+4])


        # print('gates_it', gates[0][0:4], gates_normal[0][0:4])
        # print('gates_ft', gates[0][HS:HS+4], gates_normal[0][HS:HS+4])
        # print('gates_gt', gates[0][HS * 2:HS * 2+4], gates_normal[0][HS * 2:HS * 2+4])
        # print('gates_ot', gates[0][HS * 3:HS * 3+4], gates_normal[0][HS * 3:HS * 3+4])


        gates_it = gates[:, :HS].numpy()/4
        gates_it = bip(gates_it)
        i_t = bip(usn_actual_value(tanh_activation(gates_it)))
        
        gates_ft = gates[:, HS:HS * 2].numpy()/4
        gates_ft = bip(gates_ft)
        f_t = bip(usn_actual_value(tanh_activation(gates_ft)))
        
        gates_gt = gates[:, HS * 2:HS * 3].numpy()/2
        gates_gt = bip(gates_gt)
        g_t = (tanh_activation(gates_gt))
        
        gates_ot = gates[:, HS * 3:].numpy()/4
        gates_ot = bip(gates_ot)
        o_t = bip(usn_actual_value(tanh_activation(gates_ot)))
        # print('o_t', o_t.shape)

        i_t_n, f_t_n, g_t_n, o_t_n = (
            torch.sigmoid(gates_normal[:, :HS]*2),  # input
            torch.sigmoid(gates_normal[:, HS:HS * 2]*2),  # forget
            torch.tanh(gates_normal[:, HS * 2:HS * 3]*2),
            torch.sigmoid(gates_normal[:, HS * 3:]*2),  # output
        )

        it_mse = ((i_t_n - bsn_actual_value(i_t)) ** 2).mean()
        gt_mse = ((g_t_n - bsn_actual_value(g_t)) ** 2).mean()
        ft_mse = ((f_t_n - bsn_actual_value(f_t)) ** 2).mean()
        ot_mse = ((o_t_n - bsn_actual_value(o_t)) ** 2).mean()
        # print('mse (it,  gt, ft, ot)', it_mse, gt_mse, ft_mse, ot_mse)


        # print('it', bsn_actual_value(i_t[0][0:4]), i_t_n[0][0:4])
        # print('ft', bsn_actual_value(f_t[0][0:4]), f_t_n[0][0:4])
        # print('gt', bsn_actual_value(g_t[0][0:4]), g_t_n[0][0:4])
        # print('ot', bsn_actual_value(o_t[0][0:4]), o_t_n[0][0:4])


        cx1 = ~(f_t ^ bip(cx.numpy()))
        cx2 = ~(i_t ^ g_t)

        # cx = ssum(cx1,cx2)    # SC sum using MUX
        cx = bsn_apc_2in_sum(cx1, cx2)  # SC sum using APC
        
        cx_n = (f_t_n * cx_n + i_t_n * g_t_n)
        # print('cx', bsn_actual_value(cx[0][0:4]), cx_n[0][0:4])
        
        tan = tanh_activation(~(cx ^ bip(0.5)))
        # print('tan_cx', bsn_actual_value(tan[0][0:4]), torch.tanh(cx_n)[0][0:4])

        hx = ~(o_t ^ tan)
        hx = torch.Tensor(bsn_actual_value(hx))
        cx = torch.Tensor(bsn_actual_value(cx))
        # With sample-wise normalization:
        # maximum = torch.max(torch.abs(cx), dim=-1, keepdim=True)[0]
        # maximum = torch.clamp(maximum, min=1.0)
        # # print('max',maximum)

        # cx = torch.div(cx, maximum)


        hx_n = o_t_n * torch.tanh(cx_n)

        # With sample-wise normalization:
        # maximum_n = torch.max(torch.abs(cx_n), dim=-1, keepdim=True)[0]
        # maximum_n = torch.clamp(maximum_n, min=1.0)
        # # print('max',maximum_n)

        # cx_n = torch.div(cx_n, maximum_n)
        cx_n = torch.div(cx_n, 2)

        cx_mse = ((cx_n - cx) ** 2).mean()
        hx_mse = ((hx_n - hx) ** 2).mean()
        # print('cx', cx[0][0:4], cx_n[0][0:4])
        # print('hx', hx[0][0:4], hx_n[0][0:4])
        # print('mse (cx, hx)', cx_mse, hx_mse)
        hidden_seq.append(hx.unsqueeze(0))
        hidden_seq_n.append(hx_n.unsqueeze(0))
    hidden_seq = torch.cat(hidden_seq, dim=0)
    hidden_seq_n = torch.cat(hidden_seq_n, dim=0)
    return (hidden_seq,hx,cx),(hidden_seq_n,hx_n,cx_n)



def lstm(input, lstm_size, hx, cx, weight_ih, weight_hh, bias_ih, bias_hh):
    '''
    :param input: input
    :param lstm_size: hidden layer lize
    :param hx: hidden cell state 1
    :param cx: hidden cell state 2
    :param weight_ih: input weight
    :param weight_hh: hidden layer weight
    :param bias_ih: input bias
    :param bias_hh: hidden layer bias
    :return: hx

    '''

    # type: (Tensor, Tuple[Tensor, Tensor], Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    # print(input.size())


    seq_sz, bs, _ = input.size()
    hidden_seq = []
    
    hx = hx[0, :, :]
    cx = cx[0, :, :]

    HS = lstm_size
    for t in range(seq_sz):
        maximum = 1
        x_t = input[t, :, :]
        
        # batch the computations into a single matrix multiplication
        gates = x_t @ weight_ih + hx @ weight_hh + bias_ih + bias_hh

        i_t, f_t, g_t, o_t = (
            torch.sigmoid(gates[:, :HS]*2),  # input
            torch.sigmoid(gates[:, HS:HS * 2]*2),  # forget
            torch.tanh(gates[:, HS * 2:HS * 3]*2),
            torch.sigmoid(gates[:, HS * 3:]*2),  # output
        )

        cx = f_t * cx + i_t * g_t
        hx = o_t * torch.tanh(cx)
        # With sample-wise normalization:
        # maximum = torch.max(torch.abs(cx), dim=-1, keepdim=True)[0]
        # maximum = torch.clamp(maximum, min=1.0)
                
        # cx = torch.div(cx, maximum)
        cx = torch.div(cx, 2)
        
        hidden_seq.append(hx.unsqueeze(0))
    hidden_seq = torch.cat(hidden_seq, dim=0)
    return hidden_seq


def predict(from_scratch, test_loader, model, hidden_dim,hidden_dim2, batch_size):
    dictionary = {}
    test_losses = []
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    count = 0
    new = list(model.state_dict().items())
    # print(new)
    # my_model_kvpair = net.state_dict()
    for key, value in new:
        layer_name, weights = new[count]
        dictionary[layer_name] = weights
        # mymodel_kvpair[key] = weights
        count += 1
    if from_scratch == False:
        model.eval()

        test_loss = 0
        correct = 0
        # Calculate test accuracy
        test_acc = 0.0
        for i, data in enumerate(test_loader, 0):
            print('images', i)

            inputs, labels = data
            print('images1', inputs)
            inputs = inputs.view(-1, 28, 28)
            print('images2', inputs)

            outputs = model(inputs)

            test_acc += get_accuracy(outputs, labels, batch_size)

        print('Test Accuracy: %.2f' % (test_acc / i))


    else:
        model.eval()
        correct = 0
        # Calculate test accuracy
        test_acc = 0.0
        test_loss = 0
        correct = 0
        correct_norm = 0
        test_loss_normal = 0
        maximum_w=1.0
        maximum_u=1.0
        maximum_b_w = 1.0
        maximum_b_u = 1.0
        maximum2_u = 1.0
        maximum2_w = 1.0
        for w in dictionary['lstm1.bias_w']:
            maximum_w = max(torch.max(torch.abs(w)), maximum_w)
        # print('bias_w', maximum_w)

        for u in dictionary['lstm1.bias_u']:
            maximum_u = max(torch.max(torch.abs(u)), maximum_u)
        # print('bias_u', maximum_u)

        for u in dictionary['lstm2.U']:
            maximum2_u = max(torch.max(torch.abs(u)), maximum2_u)
        # print('max_u', maximum2_u)

        for u in dictionary['lstm2.W']:
            maximum2_w = max(torch.max(torch.abs(u)), maximum2_w)
        # print('max_w', maximum2_w)

        for w in dictionary['FC.bias']:
            maximum_b_w = max(torch.max(torch.abs(w)), maximum_b_w)
        # print('lin_b', maximum_b_w)

        for u in dictionary['FC.weight']:
            maximum_b_u = max(torch.max(torch.abs(u)), maximum_b_u)
        # print('lin_', maximum_b_u)


        for i, data in enumerate(testloader, 0):
            print('no_of_images', i)
            inputs, labels = data
            print('images1', inputs.shape)
            inputs = inputs.view(-1, 28, 28)
            print('images2', inputs.shape)


            with torch.no_grad():
                inputs = inputs.permute(1, 0, 2)
                state_h1, state_c1 = (torch.zeros(1,batch_size, hidden_dim),
                                    torch.zeros(1,batch_size, hidden_dim))

                state_h2, state_c2 = (torch.zeros(1,batch_size, hidden_dim2),
                                        torch.zeros(1,batch_size, hidden_dim2))

                state_hn1, state_cn1 = (torch.zeros(1,batch_size, hidden_dim),
                                        torch.zeros(1,batch_size, hidden_dim))

                state_hn2, state_cn2 = (torch.zeros(1,batch_size, hidden_dim2),
                                        torch.zeros(1,batch_size, hidden_dim2))
              
                (embed1,state_h1,state_c1),(embed_n1,state_hn1,state_cn1)= lstm_stoc_activation(inputs,inputs, hidden_dim, state_h1, state_c1,
                                             state_hn1, state_cn1,
                                             dictionary['lstm1.W'],
                                             dictionary['lstm1.U'], dictionary['lstm1.bias_w'],
                                             dictionary['lstm1.bias_u'])
                
                # (embed2,state_h2,state_c2),(embed_n2,state_hn2,state_cn2)= lstm_stoc_activation(embed1,embed_n1, hidden_dim2, state_h2, state_c2,
                #                              state_hn2, state_cn2,
                #                              dictionary['lstm2.W'],
                #                              dictionary['lstm2.U'], dictionary['lstm2.bias_w'],
                #                              dictionary['lstm2.bias_u'])

                embed2 = lstm(embed1, hidden_dim2, state_h2, state_c2,
                                                               dictionary['lstm2.W'],
                                                               dictionary['lstm2.U'], dictionary['lstm2.bias_w'],
                                                               dictionary['lstm2.bias_u'])
                
                embed_n2 = lstm(embed_n1, hidden_dim2, state_hn2, state_cn2,
                                                               dictionary['lstm2.W'],
                                                               dictionary['lstm2.U'], dictionary['lstm2.bias_w'],
                                                               dictionary['lstm2.bias_u'])


                output_lin = Linear(embed2[-1, :, :], dictionary['FC.weight'], dictionary['FC.bias'])
                output_normal_lin = Linear(embed_n2[-1, :, :], dictionary['FC.weight'], dictionary['FC.bias'])
                output=torch.nn.functional.softmax(output_lin)
                output_normal = torch.nn.functional.softmax(output_normal_lin)
                output = output.view(-1, output_dim)
                output_normal = output_normal.view(-1, output_dim)
                test_loss += F.nll_loss(output, labels, size_average=False).item()
                # test_loss_normal += F.nll_loss(output_normal, labels, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(labels.data.view_as(pred)).sum()
                print('no_of_correct', correct)
                pred_norm = output_normal.data.max(1, keepdim=True)[1]
                correct_norm += pred_norm.eq(labels.data.view_as(pred_norm)).sum()
                print('no_of_correct_norm', correct_norm)
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        # test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        # test_acc += get_accuracy(output, labels, batch_size)

        # print('Test Accuracy: %.2f' % (test_acc / i))

    # actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:]).reshape(-1, 1))
    return None

if __name__ == "__main__":
    # Importing the dataset
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
    layer_dim = 2
    output_dim = 10
    seq_dim = 28
    num_epochs = 20
    # print(train_loader)
    model = main(True, trainloader, num_epochs, batch_size, seq_dim, input_dim, hidden_dim, hidden_dim2, output_dim,
                 layer_dim)

    count = 0
    dictionary = {}
    new = list(model.state_dict().items())
    # print(new)
    for key, value in new:
        layer_name, weights = new[count]
        dictionary[layer_name] = weights
        count += 1

    #print('size',dictionary['lstm1.bias_u'].shape)
    accuracy = predict(True, testloader, model, hidden_dim, hidden_dim2, batch_size_test)
