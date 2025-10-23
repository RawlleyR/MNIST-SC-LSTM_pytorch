# -*- coding: utf-8 -*-
"""
This script re-implements a stochastic computing in PyTorch.
It converts the original NumPy-based logic to work with PyTorch tensors,
allowing for GPU acceleration.
"""

import torch
import math

# Use CUDA if available, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Nbits = 512
Nbytes = Nbits // 8

def packbits_torch(tensor):
    """
    Implements the functionality of numpy.packbits for PyTorch tensors.
    Packs a boolean/uint8 tensor of bits into a uint8 tensor of bytes.
    """
    # Pad the tensor so its last dimension is a multiple of 8
    pad_bits = (8 - (tensor.shape[-1] % 8)) % 8
    if pad_bits > 0:
        padding = torch.zeros(tensor.shape[:-1] + (pad_bits,),
                              dtype=tensor.dtype, device=device)
        tensor = torch.cat([tensor, padding], dim=-1)
        
    # Reshape to groups of 8 bits
    tensor = tensor.reshape(tensor.shape[:-1] + (-1, 8))
    
    # Create the powers of 2 for bit-to-byte conversion
    powers_of_2 = 2 ** torch.arange(7, -1, -1, dtype=torch.uint8, device=device)
    
    # Multiply each bit by its corresponding power of 2 and sum
    return torch.sum(tensor * powers_of_2, dim=-1).to(torch.uint8)

def unpackbits_torch(tensor):
    """
    Implements the functionality of numpy.unpackbits for PyTorch tensors.
    Unpacks a uint8 tensor of bytes into a boolean tensor of bits.
    """
    # Create the powers of 2 for bit extraction
    powers_of_2 = 2 ** torch.arange(7, -1, -1, dtype=torch.uint8, device=device)
    
    # Use broadcasting to extract each bit and reshape
    return ((tensor.unsqueeze(-1) & powers_of_2) > 0).flatten(start_dim=-2)

def uni(p, n=Nbits):
    """
    Generates a unipolar stochastic bit stream.
    The input `p` is a value in [0, 1].

    Args:
        p (torch.Tensor or float): The probability of a 1.
        n (int): The length of the bit stream.

    Returns:
        torch.Tensor: A packed uint8 tensor representing the bit stream.
                      Shape: (..., Nbytes)
    """
    if not isinstance(p, torch.Tensor):
        p = torch.tensor(p, dtype=torch.float32, device=device)

    if p.dim() > 0:
        p_flat = p.flatten()
        result_list = [packbits_torch((torch.rand((n,), device=device) < val).to(torch.uint8)) for val in p_flat]
        result_tensor = torch.stack(result_list, dim=0).reshape(p.shape + (n // 8,))
        return result_tensor
    else:
        bool_stream = (torch.rand((n,), device=device) < p)
        return packbits_torch(bool_stream.to(torch.uint8))

def bip(v, n=Nbits):
    """
    Generates a bipolar stochastic bit stream.
    The input `v` is a value in [-1, 1].

    Args:
        v (torch.Tensor or float): The bipolar value.
        n (int): The length of the bit stream.

    Returns:
        torch.Tensor: A packed uint8 tensor representing the bit stream.
                      Shape: (..., Nbytes)
    """
    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v, dtype=torch.float32, device=device)
    

    return uni((v + 1) / 2, n)

def usn_actual_value(us):
    """
    Converts a unipolar bit stream back to its numerical value.

    Args:
        us (torch.Tensor): A packed uint8 tensor representing the bit stream.

    Returns:
        torch.Tensor: The numerical value in [0, 1].
    """
    # Unpack the tensor to get the individual bits
    bool_stream = unpackbits_torch(us)
    # The value is the mean of the bits (treated as floats)
    return torch.mean(bool_stream.float(), dim=-1)

def bsn_actual_value(bs):
    """
    Converts a bipolar bit stream back to its numerical value.

    Args:
        bs (torch.Tensor): A packed uint8 tensor representing the bit stream.

    Returns:
        torch.Tensor: The numerical value in [-1, 1].
    """
    # Unpack the tensor and convert the unipolar value back to the bipolar value
    return usn_actual_value(bs) * 2 - 1

# def tanh_activation(sn):
#     """
#     Args:
#         sn (torch.Tensor): A packed uint8 tensor representing the input bit stream.

#     Returns:
#         torch.Tensor: A packed uint8 tensor representing the output bit stream.
#     """
#     # Unpack the input tensor to get individual bits
#     bool_stream = unpackbits_torch(sn)
#     # print("bool shape: ", bool_stream.size())
    
#     # The state is a tensor, one for each input value
#     state = torch.full(bool_stream.shape[:-1], 3, dtype=torch.long, device=device)
#     # print("state shape:", state.size())
    
#     # Transpose the tensor to iterate over bits (last dimension)
#     bool_stream_t = torch.movedim(bool_stream,-1, 0)
#     # print("transpose shape: ",bool_stream_t.size())
#     out = torch.empty_like(bool_stream_t)

#     # Iterate bit by bit to simulate the state machine
#     for i, bit_slice in enumerate(bool_stream_t):
#         out[i] = state > 3
#         bit_as_int = bit_slice.long()
#         state = state + (bit_as_int * (state < 7)) - ((1 - bit_as_int) * (state > 0))
    
#     # Transpose back to original shape and pack the output tensor
#     return packbits_torch(out.movedim(0, out.ndim-1).to(torch.uint8))

def tanh_activation(sn):
    # function to maximize GPU utilization
    
    bool_stream = unpackbits_torch(sn).to(device)
    bit_stream = bool_stream.long()
    state = torch.full(bool_stream.shape[:-1], 3, dtype=torch.long, device=device)
    out = []

    for i in range(bool_stream.shape[-1]):
        state = torch.where(
            bit_stream[..., i] == 1,
            torch.minimum(state + 1, torch.tensor(7, device=device)),
            torch.maximum(state - 1, torch.tensor(0, device=device))
        )
        out.append((state > 3).to(torch.uint8))

    out = torch.stack(out, dim=-1)
    return packbits_torch(out)

def ssum(a, b):
    """
    Stochastic addition using a multiplexer.

    Args:
        a (torch.Tensor): A packed uint8 stochastic bit stream.
        b (torch.Tensor): A packed uint8 stochastic bit stream.

    Returns:
        torch.Tensor: The resulting sum packed uint8 bit stream.
    """
    # Unpack input tensors
    a_unpacked = unpackbits_torch(a)
    b_unpacked = unpackbits_torch(b)
    
    # Create a random selection bit stream with 0.5 probability
    c = uni(0.5)
    c_unpacked = unpackbits_torch(c)
    
    # Create the inverted selection bit stream
    d_unpacked = ~c_unpacked
    
    # Implement the MUX operation: (a AND c) OR (b AND d)
    out_unpacked = (a_unpacked & c_unpacked) | (b_unpacked & d_unpacked)
    return packbits_torch(out_unpacked.to(torch.uint8))

