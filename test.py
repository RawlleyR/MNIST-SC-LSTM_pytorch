from stochastic import *
from apc import bsn_apc_2in_sum

a = np.array([0.3, 0.7])
b = np.array([0.4, 0.5])

a_bsn = bip(a)
b_bsn = bip(b)

bsn_sum = bsn_apc_2in_sum(a_bsn, b_bsn)

print('actual sum: ', a+b)
print('bsn sum: ', bsn_actual_value(bsn_sum))