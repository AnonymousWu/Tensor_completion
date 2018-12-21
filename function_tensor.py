import ctf,time,random
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from ctf import random as crandom
glob_comm = ctf.comm()

def function_tensor(I,J,K,sparsity):
    #N = 5
    #n = 51
    #L = 100
    #nsample = 10*N*n*L #10nNL = 255000
    n = 10

    v = np.linspace(-1,1,n)
    #v = np.arange(1,n+1)
    v = ctf.astensor(v**2)
    
    v2 = ctf.tensor(n,sp=True)
    v2 = v

    
    T = ctf.tensor((I,J,K),sp=True)
    T.fill_sp_random(1,1,sparsity)
    #T = ctf.exp(-1 * ctf.power(ctf.power(T,2),0.5))  # x = exp(-sqrt(x^2)) 

    T2 = ctf.tensor((I,J,K),sp=True)
    T2.i("ijk") << T.i("ijk") * v2.i("i")
    T2.i("ijk") << T.i("ijk") * v2.i("j")
    T2.i("ijk") << T.i("ijk") * v2.i("k")
    
    T2 = ctf.power(T2,0.5)
    T2 = (-1.0) * T2
    
    #T2 = ctf.exp(T2)
    
    return T2