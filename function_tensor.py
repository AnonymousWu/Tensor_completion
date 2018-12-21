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
    
    T = ctf.tensor((I,J,K),sp=True)
    T.fill_sp_random(-1,1,sparsity)
    T = ctf.exp(-1 * ctf.power(ctf.power(T,2),0.5))  # x = exp(-sqrt(x^2)) 
    
    return T
    