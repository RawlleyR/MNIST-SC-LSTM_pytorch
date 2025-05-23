import numpy as np

from numpy import random
import math
from scipy.special import expit, logit
Nbits = 512
import time
Nbytes = Nbits // 8

def uni(p, n=Nbits):
    if isinstance(p, list):
        return uni(np.array(p), n)
    if isinstance(p, np.ndarray):
        return np.array([uni(v, n) for v in p.flat]).reshape(p.shape + (n // 8,))

    var_x = np.packbits((np.random.rand(n) < p).astype(int))
    return var_x

def bip(v, n=Nbits):
    if isinstance(v, list):
        return bip(np.array(v), n)
    usn_value = uni((v + 1) / 2, n)
    return usn_value

_usn_value = np.asarray([bin(x).count('1') / 8 for x in range(256)])
_bsn_value = _usn_value * 2 - 1





def usn_actual_value(us):
    return np.average(_usn_value[us], axis=-1)


def bsn_actual_value(bs):
    return np.average(_bsn_value[bs], axis=-1)
#print(bsn_actual_value(value),x_nt)


def sn_tanh_slow(sn):
    x = np.unpackbits(sn)
    state = 3  # starting state
    for j, bit in enumerate(x):
        x[j] = state > 3
        if state < 7 and bit:
            state += 1
        if state > 0 and not bit:
            state -= 1
    return np.packbits(x)


def _sn_tanh(byte, start_state):
    x = np.unpackbits(np.array([byte], dtype='uint8'))
    out = np.empty_like(x)
    state = start_state
    for j, bit in enumerate(x):
        x[j] = state > 3
        if state < 7 and bit:
            state += 1
        if state > 0 and not bit:
            state -= 1

    #print('jgg', np.packbits(x)[0])
    return (np.packbits(x)[0], state)

for state in range(8):
    for byte in range(256):
        _sn_tmp= _sn_tanh(byte, state)
        #print(_sn_tmp)
_sn = np.array([[_sn_tanh(byte, state) for byte in range(256)] for state in range(8)])
#print('len',_sn_tmp[0])
_sn_tanh_out = _sn[:, :, 0]
#print('3',_sn_tanh_out[3])
_sn_tanh_state = _sn[:, :, 1]

#print('jgg',_sn_tanh_lookup_state)
def tanh_activation(sn):
    if sn.ndim > 1:
        state = np.zeros(sn.shape[:-1], dtype='uint8') + 3
    else:
        state = 3
    sn = np.rollaxis(sn, -1)
    out = np.empty_like(sn)
    for i, byte in enumerate(sn):
        out[i] = _sn_tanh_out[state, byte]
        state = _sn_tanh_state[state, byte]

    #print(out)
    xx=np.packbits(out)
    #print('pack',xx)
    return np.rollaxis(out, 0, out.ndim)
'''
cc=tanh_activation(value)
#print('fah',cc)
layer1=bsn_actual_value(cc)
#print(layer1)
ccc=np.unpackbits(cc)
#print('guv',ccc)
tannnn=sng_main.tanh_activation(x_n,128)
#print('bkhk',tannnn)
tanh=sng_main.stochastic2bin(tannnn,1024,True)
#print(layer1,tanh)
'''
def random_choice(choices=2, n=Nbits):
    r = np.random.randint(0, choices, n)
    return np.array([np.packbits((r == i).astype(int)) for i in range(choices)])

# Summation
def s_sum(a):

    s = random_choice(a.shape[0])
    sum = np.zeros(a.shape[1:], dtype='uint8')
    for i in range(a.shape[0]):
        sum |= a[i] & s[i]
    return sum

# Summation
def s_sum_alt(a):
    #print('a1',a.shape[0])
    sum = np.zeros(a.shape[1:], dtype='uint8')
    if a.shape[0]%2!=0:
        a1=a[0:math.floor(a.shape[0]/2)]
        #print('a1',a1)
        a2=a[math.floor(a.shape[0]/2):a.shape[0]-1]
        a3=a[a.shape[0]-1]
        b = ssum(a1, a2)
        return (np.append(b,[a3],axis = 0))
    else:
        a1 = a[0:math.floor(a.shape[0] / 2)]
        a2 = a[math.floor(a.shape[0] / 2):a.shape[0]]
        return (ssum(a1, a2))


def summ(a):
    while a.shape[0]!=1:
        a=s_sum_alt(a)

    return a[0]


def ssum(a,b):
    a=uni(usn_actual_value(a))
    b=uni(usn_actual_value(b))
    #print('add',(bsn_actual_value(a)+bsn_actual_value(b))/2)
    #c=np.full(a.shape, 0.5)
    #print('c',c)
    c=uni(0.5)
    #print('c', bsn_actual_value(c))
    d=~(c)
    #print('d', bsn_actual_value(d))
    #print('a', bsn_actual_value(~(b^d)))
    #print('b', bsn_actual_value(a&c))
    #out = ((bsn_actual_value(a)+bsn_actual_value(b))/2)
    out=(a&c)|(b&d)
    out=(usn_actual_value(out)*2)-1
    #out1=a|b
    #out1=(~(a|b)^c)
    #out1=((a|b)&c)
    #out = bsn_actual_value(out1)
    #print('Add_og',bsn_actual_value(out))
    return bip(out)

def mat_m(M, N):
    # List to store matrix multiplication result
    x,y,_=M.shape
    m,n,_=N.shape
    R = np.zeros((x, n,64), dtype='uint8')


    #R=bip(R)
    #print('R', R)
    #R = np.zeros((x, n))

    for i in range(M.shape[0]):
        for j in range(N.shape[1]):
            for k in range(N.shape[0]):
                #print('M',np.unpackbits(M[i][k]))
                R[i][j] = ssum((R[i][j]), (~(M[i][k] ^ N[k][j])))

#    for i in range(M.shape[0]):
 #       for j in range(N.shape[1]):
            # if we use print(), by default cursor moves to next line each time,
            # Now we can explicitly define ending character or sequence passing
            # second parameter as end="<character or string>"
            # syntax: print(<variable or value to print>, end="<ending with>")
            # Here space (" ") is used to print a gape after printing
            # each element of R
  #          print(R[i][j], end=" ")
   #     print("\n", end="")
    return R
    # First matrix. M is a list


def mat_m_alt(M, N):
    # List to store matrix multiplication result
    x,y,_=M.shape
    m,n,_=N.shape
    R = np.zeros((x, m,64), dtype='uint8')


    #R=bip(R)
    #print('R', R)
    #R = np.zeros((x, n))

    for i in range(M.shape[0]):
        for j in range(N.shape[0]):

                #print('M',np.unpackbits(M[i][k]))

                R[i][j] = s_sum(~(M[i] ^ N[j]))
#    for i in range(M.shape[0]):
 #       for j in range(N.shape[1]):
            # if we use print(), by default cursor moves to next line each time,
            # Now we can explicitly define ending character or sequence passing
            # second parameter as end="<character or string>"
            # syntax: print(<variable or value to print>, end="<ending with>")
            # Here space (" ") is used to print a gape after printing
            # each element of R
  #          print(R[i][j], end=" ")
   #     print("\n", end="")
    return R
    # First matrix

def sum_complex(Real, Imaginary):
    R1, R2 = Real
    I1, I2 = Imaginary
    Real = ssum(R1, R2)
    Imaginary = ssum(I1, I2)
    return [Real, Imaginary]


def mul_complex(Real, Imaginary):
    R1, R2 = Real
    I1, I2 = Imaginary
    Real = ssum((~(R1 ^ R2)), bip(-(bsn_actual_value(~(I1 ^ I2)))))
    Imaginary = ssum((~(R1 ^ I2)), (~(I1 ^ R2)))

    return [Real, Imaginary]

def element_wise_mult(M, N):
    # List to store matrix multiplication result
    M=bip(M)
    N=bip(N)
    x,y,_=M.shape
    m,n,_=N.shape
    R = np.zeros((x, y,128), dtype='uint8')


    #R=bip(R)
    #print('R', R)
    #R = np.zeros((x, n))

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            R[i][j] =(~(M[i][j] ^ N[i][j]))

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            # if we use print(), by default cursor moves to next line each time,
            # Now we can explicitly define ending character or sequence passing
            # second parameter as end="<character or string>"
            # syntax: print(<variable or value to print>, end="<ending with>")
            # Here space (" ") is used to print a gape after printing
            # each element of R
            print(R[i][j], end=" ")
        print("\n", end="")
    return R

'''
arr1= np.random.rand(12, 1)
#print('arra1',arr1)
arr1=bip(arr1)
#print(arr1.shape)
arr2 = np.random.rand(1, 400)
arr2=bip(arr2)

mat_m(arr1,arr2)




a=bip(0.22)
b=(~(a^bip(0.5)))
d=0.85
#b=bip(d/2)
c=tanh_activation(a)
print('tan',bsn_actual_value(c),np.tanh(0.22))
print('sig',usn_actual_value(c), 1/(1 + np.exp(-0.22)) )
b=(~(bip(d)^bip(0.5)))
c=tanh_activation(b)
print('sigmoid2',usn_actual_value(c))

ssum(a,b)
'''
def mult(a,b):
    #x = np.unpackbits(a)
    #y = np.unpackbits(b)
    #print(x)
    #out = np.zeros(x.shape, dtype='uint8')
    out=[]
    out=bsn_actual_value(~(a ^ b))


    return out

# x=-0.11
# x1=(x+1)/2
# sigvalue=~(bip(x)^bip(0.5))
# #sigvalue=~(bip(x)^bip(0.5))
# value=uni(x1)
# print('fah',usn_actual_value(bip(x)))
# cc=tanh_activation(value)
# value2=(x+1)/2
# cc_sig=tanh_activation(bip(x/2))
# print('fah',cc)
# layer1=usn_actual_value(cc)
# print('tanh',2*usn_actual_value(cc)-1,np.tanh(x))
# print('tanh',bsn_actual_value(cc_sig)/2,np.tanh(-9.6))
# print('sigmoid',usn_actual_value(cc_sig),expit(-2.94))

# print('bip',bsn_actual_value(bip(bsn_actual_value(tanh_activation(bip(0.267))))))
# print('uni',bsn_actual_value(bip(usn_actual_value(tanh_activation(bip(0.267))))))

# array_A = random.random(size=(5,28))
# input_A = random.random(size=(28, 1600))

# start = time.time()
# print('start',start)
# print("hello")

# array_out=mat_m_alt(bip(array_A),bip(np.transpose(input_A)))
# end = time.time()
# print('end',end)
# print(end - start)
# normal=(array_A@ input_A)/28
# print('normal',(array_A@ input_A)/28)
# print('stochastic', bsn_actual_value(array_out))
# print('mse',((normal - bsn_actual_value(array_out)) ** 2).mean())

# x=bip(0.0)
# print('x',bsn_actual_value(x))
# y=bip(0.50)
# print('y',bsn_actual_value(y))
# z=(~(x ^ y))
# print('z',bsn_actual_value(z))

# #print(bsn_actual_value(ssum(bip(0.58),bip(-(bsn_actual_value(z))))))
# a=bip(0.519)
# print('t',usn_actual_value(a))
# print(bsn_actual_value(bip(0.613)))
# b=bip(-bsn_actual_value(bip(0.613)))
# print('r',usn_actual_value(b))
# #print('sum',(usn_actual_value(ssum(uni(usn_actual_value(a)),uni(usn_actual_value(b))))*2)-1)
# #print('sum1',(bsn_actual_value(ssum(uni(usn_actual_value(a)),uni(usn_actual_value(b))))))
# print('sum2',(bsn_actual_value(ssum(a,b))))

# print('mult1',mult(a,b))
# #print('mult2',1.*0.8*-0.79)


# #'''








