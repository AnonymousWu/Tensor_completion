import ctf, time, random, sys
import numpy as np
from functools import reduce
import numpy.linalg as la
from ctf import random as crandom
import gzip
import shutil
import os

sys.path.insert(0, '../SGD')
from gradient1 import sparse_SGD
sys.path.insert(0, '../ALS')
from als_sp import getALS_CG
sys.path.insert(0, '../CCD')
#from ccd_sp import

glob_comm = ctf.comm()

def getOmega(T):
    [inds, data] = T.read_local_nnz()
    data[:] = 1.
    Omega = ctf.tensor(T.shape,sp=True)
    Omega.write(inds,data)
    return Omega

modify = False
def read_from_frostt(file_name, I, J, K):
    unzipped_file_name = file_name + '.tns'
    exists = os.path.isfile(unzipped_file_name)

    if not exists:
        if glob_comm.rank() == 0:
            print('Creating ' + unzipped_file_name)
        with gzip.open(file_name + '.tns.gz', 'r') as f_in:
            with open(unzipped_file_name, 'w') as f_out:
                shutil.copyfileobj(f_in, f_out)

    T_start = ctf.tensor((I + 1, J + 1, K + 1), sp=True)
    if glob_comm.rank() == 0:
        print('T_start initialized')
    T_start.read_from_file(unzipped_file_name)
    if glob_comm.rank() == 0:
        print('T_start read in')
    T = ctf.tensor((I, J, K), sp=True)
    if glob_comm.rank() == 0:
        print('T initialized')
    T[:, :, :] = T_start[1:, 1:, 1:]
    if glob_comm.rank() == 0:
        print('T filled')

    if modify:
        T.write_to_file(unzipped_file_name)

    return T

def creat_perfect_tensor(I, J, K, r, sparsity):
    U = ctf.random.random((I, r))
    V = ctf.random.random((J, r))
    W = ctf.random.random((K, r))
    T = ctf.tensor((I, J, K), sp=True)
    T.i("ijk") << U.i("ir") * V.i("jr") * W.i("kr")
    T.sample(sparsity)
    return T

def main():

    # parameters that must be passed in 
    I = int(sys.argv[1])
    J = int(sys.argv[2])
    K = int(sys.argv[3])
    sparsity = np.float64(sys.argv[4])
    r = int(sys.argv[5])
    
    if ctf.comm().rank() == 0:
        print("I is",I,"J is",J,"K is",K,"sparisty is",sparsity,"r is",r)

    # parameters that are optional with default value
    regParam = 0.1
    if len(sys.argv) >= 7:
        regParam= np.float64(sys.argv[6])
    
    # SGD parameters
    sgd_stepSize = 0.01
    if len(sys.argv) >= 8:
        sgd_stepSize = float(sys.argv[7])

    sgd_sample_rate = 0.01
    if len(sys.argv) >= 9:
        sgd_sample_rate = float(sys.argv[8])

    # ALS parameters
    block_size = int(sys.argv[1])   # same as input I size
    if len(sys.argv) >= 10:
        block_size = int(sys.argv[9])

    use_func = 0
    if len(sys.argv) >= 11:
        use_func= int(sys.argv[10])

    num_iter = 1
    if len(sys.argv) >= 12:
        num_iter = int(sys.argv[11])

    err_thresh = .001
    if len(sys.argv) >= 13:
        err_thresh = np.float64(sys.argv[12])

    run_implicit = 1
    if len(sys.argv) >= 14:
        run_implicit= int(sys.argv[13])

    run_explicit = 1
    if len(sys.argv) >= 15:
        run_implicit= int(sys.argv[14])
    
    
    #generate tensor
    T = ctf.tensor((I, J, K), sp=True)
    T.fill_sp_random(0., 1., sparsity)
    # T.read_from_file("T.txt")
    
    omega = getOmega(T)
    s = T.sum()
    os = omega.sum()
    if ctf.comm().rank() == 0:
        print("T sum is", s, "Omega sum is", os)
    
    U = ctf.random.random((I, r))
    V = ctf.random.random((J, r))
    W = ctf.random.random((K, r))

    #SGD CODE
    if ctf.comm().rank() == 0:
        print("Begin SGD")
        print("step_size is", sgd_stepSize, "sample_rate is", sgd_sample_rate)
    
    U_SGD = ctf.copy(U)
    V_SGD = ctf.copy(V)
    W_SGD = ctf.copy(W)
    T_SGD = ctf.copy(T)
    sparse_SGD(T_SGD, U_SGD, V_SGD, W_SGD, regParam, omega, I, J, K, r, sgd_stepSize, sgd_sample_rate)
    
    if ctf.comm().rank() == 0:
        print("End SGD")

    #ALS CODE
    if ctf.comm().rank() == 0:
        print("Begin ALS")
        print("block_size is",block_size, "use_func is",use_func,"num_iter is",num_iter,"err_thresh is",err_thresh,"run_implicit",run_implicit,"run_explicit is",run_explicit)
    
    U_CG1 = ctf.copy(U)
    V_CG1 = ctf.copy(V)
    W_CG1 = ctf.copy(W)
    T_CG1 = ctf.copy(T)

    U_CG2 = ctf.copy(U)
    V_CG2 = ctf.copy(V)
    W_CG2 = ctf.copy(W)
    T_CG2 = ctf.copy(T)
    if run_implicit == True:
        getALS_CG(T_CG1,U_CG1,V_CG1,W_CG1,regParam,omega,I,J,K,r,block_size,num_iter,err_thresh,True) 
    if run_explicit == True:
        getALS_CG(T_CG2,U_CG2,V_CG2,W_CG2,regParam,omega,I,J,K,r,block_size,num_iter,err_thresh,False)

    if ctf.comm().rank() == 0:
        print("End ALS")

    #CCD CODE

main()
