import torch
import torch.nn as nn
from stochastic_torch import *
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch
import foolbox as fb
from tqdm import tqdm


def half_adder(a, b):
    """
    Performs a half-adder operation on two bit streams.
    
    Args:
        a (torch.Tensor): A boolean tensor of bits.
        b (torch.Tensor): A boolean tensor of bits.
        
    Returns:
        tuple: A tuple containing the sum and carry bit streams.
    """
    sum_ = a ^ b
    carry = a & b
    return sum_, carry

def full_adder(a, b, cin):
    """
    Performs a full-adder operation on three bit streams.
    
    Args:
        a (torch.Tensor): A boolean tensor of bits.
        b (torch.Tensor): A boolean tensor of bits.
        cin (torch.Tensor): A boolean tensor representing the carry-in.
        
    Returns:
        tuple: A tuple containing the sum and carry-out bit streams.
    """
    sum1, carry1 = half_adder(a, b)
    sum_final, carry2 = half_adder(sum1, cin)
    cout = carry1 | carry2
    return sum_final, cout

def apc_2in(input1, input2):
    """
    Approximate carry-propagation adder for two inputs.
    
    Args:
        input1 (torch.Tensor): A packed uint8 stochastic bit stream.
        input2 (torch.Tensor): A packed uint8 stochastic bit stream.
        
    Returns:
        tuple: A tuple containing the total count of 1s and the total number of bits.
    """
    # Unpack the bitstreams along the last axis
    in1_unpacked = unpackbits_torch(input1)
    in2_unpacked = unpackbits_torch(input2)

    # Determine shape and number of stages
    *shape, n = in1_unpacked.shape
    n_stages = (2 * n).bit_length()

    # Initialize accumulator sums: shape (..., n_stages)
    sums = torch.zeros(shape + [n_stages], dtype=torch.bool, device=device)
    carry_out = torch.zeros(shape, dtype=torch.bool, device=device)

    # Iterate over each bit position
    for i in range(n):
        a = in1_unpacked[..., i]
        b = in2_unpacked[..., i]

        # First HA
        sum0, carry0 = half_adder(a, b)

        # HA1
        sum1, carry1 = half_adder(sum0, sums[..., 0])
        sums[..., 0] = sum1

        # FA2: carry0, 0, carry0
        sum2, carry2 = full_adder(carry1, sums[..., 1], carry0)
        sums[..., 1] = sum2
        
        carry_next = carry2

        # Ripple through remaining stages
        for j in range(2, n_stages):
            sumj, carryj = half_adder(sums[..., j], carry_next)
            sums[..., j] = sumj
            carry_next = carryj

        carry_out = carry_next

    # Convert binary to integer
    powers = 2 ** torch.arange(n_stages, dtype=torch.float, device=device)
    sums_float = sums.float()
    
    total = torch.tensordot(sums_float, powers, dims=([-1], [0])) + (carry_out.float() * (2**n_stages))
    return total, torch.tensor(2 * n, dtype=torch.long, device=device)

def bsn_apc_2in_sum(in1, in2):
    """
    Performs stochastic addition using the approximate computing approach.
    
    Args:
        in1 (torch.Tensor): A packed uint8 bipolar stochastic bit stream.
        in2 (torch.Tensor): A packed uint8 bipolar stochastic bit stream.
        
    Returns:
        torch.Tensor: A packed uint8 bipolar stochastic bit stream representing the sum.
    """
    tot_1s_count, tot_bits = apc_2in(in1, in2)
    bsn_value = (2 * tot_1s_count.float() / tot_bits.float()) - 1
    
    bsn = bip(bsn_value)
    return bsn


def lstm_stoc_activation(input1, lstm_size, hx, cx, weight_ih, weight_hh, bias_ih, bias_hh):
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
    hx = hx[0, :, :]
    cx = cx[0, :, :]
    HS = lstm_size
    for t in range(seq_sz):
        x_t = input1[t, :, :]


        #print('shape', x_t.shape)

        maximum=1.0
        # print('shape', weight_ih.t().shape)
        # batch the computations into a single matrix multiplication
        gates = x_t @ weight_ih + hx @ weight_hh + bias_ih + bias_hh


        gates_it = gates[:, :HS]/4
        gates_it = bip(gates_it)
        i_t = bip(usn_actual_value(tanh_activation(gates_it)))
        
        gates_ft = gates[:, HS:HS * 2]/4
        gates_ft = bip(gates_ft)
        f_t = bip(usn_actual_value(tanh_activation(gates_ft)))
        
        gates_gt = gates[:, HS * 2:HS * 3]/2
        gates_gt = bip(gates_gt)
        g_t = (tanh_activation(gates_gt))
        
        gates_ot = gates[:, HS * 3:]/4
        gates_ot = bip(gates_ot)
        o_t = bip(usn_actual_value(tanh_activation(gates_ot)))
        # print('o_t', o_t.shape)


        cx1 = ~(f_t ^ bip(cx))
        cx2 = ~(i_t ^ g_t)

        # cx = ssum(cx1,cx2)    # SC sum using MUX
        cx = bsn_apc_2in_sum(cx1, cx2)  # SC sum using APC
        
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
        
        hidden_seq.append(hx.unsqueeze(0))
    hidden_seq = torch.cat(hidden_seq, dim=0)
    return hidden_seq,hx,cx



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

def Linear(x, weight, bias):
    '''
        :param x: first value to be multiplied
        :param weight: second value to be multiplied
        :param bias: value to add
        :return: actual value of a stochastic number so obtained
    '''

    out = x @ weight.t() + bias
    return out


class SCHybridModel(nn.Module):
    """
    Stochastic-first-LSTM hybrid model.
    First LSTM layer uses your PyTorch stochastic primitives (lstm_stoc_activation).
    Second LSTM layer and FC are deterministic (use your lstm and Linear).
    Returns logits (not softmax), shape [batch, n_classes].
    """
    def __init__(self, dictionary, hidden_dim, hidden_dim2, output_dim, device='cpu'):
        super().__init__()
        self.dictionary = dictionary  # mapping of weight tensors (torch.Tensor)
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2
        self.output_dim = output_dim
        self.device = torch.device(device)

        # Create a torch.nn.Linear for FC (so parameters are on correct device)
        self.fc = nn.Linear(hidden_dim2, output_dim)
        # load weights from dictionary (ensure shapes match)
        with torch.no_grad():
            self.fc.weight.copy_(dictionary['FC.weight'])
            self.fc.bias.copy_(dictionary['FC.bias'])

        # If you prefer to keep deterministic LSTM2 as custom function,
        # we will call your `lstm` helper directly in forward, using weights from dictionary.

    def forward(self, x):
        """
        x: either (N,1,28,28) or (N,28,28) — consistent with your other functions.
        Returns logits: shape (N, output_dim)
        """
        # Format input: (batch,1,28,28) -> (batch,28,28) -> permute to (seq_len, batch, features)
        if x.dim() == 4:
            x = x.squeeze(1)  # (N, 28, 28)
        # permute to (seq_len=28, batch, features=28)
        x_seq = x.permute(1, 0, 2).contiguous().to(self.device)

        batch_size = x_seq.size(1)

        state_h1 = torch.zeros(1, batch_size, self.hidden_dim, device=self.device)
        state_c1 = torch.zeros(1, batch_size, self.hidden_dim, device=self.device)

        embed1, h1, c1 = lstm_stoc_activation(
            x_seq,             # input (seq_len, batch, features)
            self.hidden_dim,
            state_h1, state_c1,
            self.dictionary['lstm1.W'].to(self.device),
            self.dictionary['lstm1.U'].to(self.device),
            self.dictionary['lstm1.bias_w'].to(self.device),
            self.dictionary['lstm1.bias_u'].to(self.device)
        )
        
        W2 = self.dictionary['lstm2.W'].to(self.device)
        U2 = self.dictionary['lstm2.U'].to(self.device)
        bw2 = self.dictionary['lstm2.bias_w'].to(self.device)
        bu2 = self.dictionary['lstm2.bias_u'].to(self.device)

        state_h2 = torch.zeros(1, batch_size, self.hidden_dim2, device=self.device)
        state_c2 = torch.zeros(1, batch_size, self.hidden_dim2, device=self.device)

        embed2 = lstm(embed1, self.hidden_dim2, state_h2, state_c2,
                      W2, U2, bw2, bu2)  # shape (seq_len, batch, hidden_dim2)

        # final linear
        logits = Linear(embed2[-1, :, :], self.fc.weight, self.fc.bias)  # shape (batch, output_dim)
        logits = logits.to(self.device)
        return logits

##-----------------------------------------------------------------------------------------------------------

def generate_boundary_adversarial_examples_gpu_autoinit(model, testloader, device,
                                                        start_batch=0, num_batches=5,
                                                        steps=5000, spherical_step=0.01,
                                                        step_adaptation=1.5):
    """
    Boundary Attack per-sample, relying on Foolbox to find initial adversarials (no seed search).
    Returns: adv_images_all, orig_images_all, predicted_on_adv, labels_all, success_flags, info
    """
    model.eval()
    model.to(device)

    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    atk = fb.attacks.BoundaryAttack(steps=steps,
                                    spherical_step=spherical_step,
                                    step_adaptation=step_adaptation)

    adv_images = []
    orig_images = []
    adv_preds = []
    labels_all = []
    success_flags = []

    dataiter = iter(testloader)
    for _ in range(start_batch):
        try:
            next(dataiter)
        except StopIteration:
            return None  # nothing to do

    total_processed = 0
    total_successes = 0
    total_init_failures = 0

    for _ in tqdm(range(num_batches), desc=f"Running Boundary Attack batches {start_batch+1}-{start_batch+num_batches}\n"):
        try:
            images, labels = next(dataiter)
        except StopIteration:
            break

        # match your CW preprocessing: remove channel dim if present
        if images.dim() == 4 and images.size(1) == 1:
            images = images.squeeze(1)  # (N, 28, 28)

        batch_size = images.size(0)
        for i in tqdm(range(batch_size), desc=f"Running per image\n"):
            print(f"Image {i}")
            total_processed += 1
            x = images[i].unsqueeze(0).to(device)   # (1, 28, 28) -> model.forward should handle permute internally
            y = labels[i].unsqueeze(0).to(device)

            try:
                raw_advs, clipped_advs, success = atk(fmodel, x, y, epsilons=None)
                succeeded = bool(success[0])
                if succeeded:
                    adv = clipped_advs[0].detach().cpu()
                    with torch.no_grad():
                        logits = model(clipped_advs.to(device))
                        pred = logits.argmax(dim=1)[0].detach().cpu()
                    adv_images.append(adv)
                    orig_images.append(x.detach().cpu()[0])
                    adv_preds.append(pred)
                    labels_all.append(y.detach().cpu()[0])
                    success_flags.append(True)
                    total_successes += 1
                else:
                    # attack ran but didn't find a valid adversarial
                    adv_images.append(clipped_advs[0].detach().cpu())
                    orig_images.append(x.detach().cpu()[0])
                    with torch.no_grad():
                        logits = model(clipped_advs.to(device))
                        pred = logits.argmax(dim=1)[0].detach().cpu()
                    adv_preds.append(pred)
                    labels_all.append(y.detach().cpu()[0])
                    success_flags.append(False)

            except ValueError:
                # init failed for this sample — record original as fallback and continue
                print(f"Init ValueError. Entered the except block for image {i}.")
                total_init_failures += 1
                success_flags.append(False)
                adv_images.append(x.detach().cpu()[0])
                orig_images.append(x.detach().cpu()[0])
                with torch.no_grad():
                    logits = model(x)
                    pred = logits.argmax(dim=1)[0].detach().cpu()
                adv_preds.append(pred)
                labels_all.append(y.detach().cpu()[0])
                continue

    # stack outputs
    adv_images_all = torch.stack(adv_images) if len(adv_images) > 0 else torch.empty(0)
    orig_images_all = torch.stack(orig_images) if len(orig_images) > 0 else torch.empty(0)
    adv_preds = torch.stack(adv_preds) if len(adv_preds) > 0 else torch.empty(0)
    labels_all = torch.stack(labels_all) if len(labels_all) > 0 else torch.empty(0)
    success_flags_tensor = torch.tensor(success_flags, dtype=torch.bool).cpu()

    info = {
        'total_processed': total_processed,
        'total_successes': total_successes,
        'total_init_failures': total_init_failures,
        'success_fraction': total_successes / max(1, total_processed)
    }

    print("Boundary Attack autoinit finished:", info)
    return adv_images_all, orig_images_all, adv_preds, labels_all, success_flags_tensor, info

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
    start_batch = 2
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

    x_adv, x_test, y_adv, y_test, success_flags, info = generate_boundary_adversarial_examples_gpu_autoinit(
    sc_model, testloader, device, start_batch, num_batches, steps, spherical_step, step_adaptation)
    
    success_frac = info['success_fraction']
    
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
        'original_labels': y_test.clone().detach().cpu(),
        'success_flags': success_flags,
        'info': info
    }, f'boundary_adv_{num_batches*batch_size_test}samples_tensorattacks_batch{start_batch+1}-{start_batch+num_batches}({success_frac*100:.2f}%_{steps}_{step_adaptation}).pt')

    print(f"Adversarial samples saved to boundary_samples.pt")