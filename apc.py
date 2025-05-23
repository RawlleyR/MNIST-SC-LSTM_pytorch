import  math
import numpy as np
from stochastic import bip, uni

def half_adder(a, b):
    sum_ = a ^ b
    carry = a & b
    return sum_, carry

def full_adder(a, b, cin):
    sum1, carry1 = half_adder(a, b)
    sum_final, carry2 = half_adder(sum1, cin)
    cout = carry1 | carry2
    return sum_final, cout

def apc_2in(input1, input2):
    """
    input1, input2: np.ndarray of shape (..., n_bytes)
        Each element is a packed stochastic bitstream
    Returns: np.ndarray of shape (...), with the total number of 1s per element
    """

    # Unpack the bitstreams along the last axis
    in1_unpacked = np.unpackbits(input1, axis=-1)
    in2_unpacked = np.unpackbits(input2, axis=-1)

    # Determine shape
    assert in1_unpacked.shape == in2_unpacked.shape
    *shape, n = in1_unpacked.shape
    n_stages = (2 * n).bit_length()

    # Initialize accumulator sums: shape (..., n_stages)
    sums = np.zeros(shape + [n_stages], dtype=np.uint8)
    carry_out = np.zeros(shape, dtype=np.uint8)

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

        carry_out = carry_next  # optional: final carry if overflow matters

    # Convert binary to integer
    powers = 2 ** np.arange(n_stages, dtype=np.uint64)
    total = np.tensordot(sums, powers, axes=([-1], [0])) + (carry_out.astype(np.uint64) << np.uint64(n_stages))
    return total, 2*n

def bsn_apc_2in_sum(in1, in2):
    tot_1s_count, tot_bits = apc_2in(in1, in2)
    bsn_value = (2*tot_1s_count/tot_bits) - 1
    # maximum = 1
    # for t in bsn_value:
    #     maximum = max(np.max(abs(t)), maximum)
    # bsn_value = bsn_value/maximum
    bsn = bip(bsn_value)
    
    return bsn
    