import ctf, time, random, sys
import numpy as np
import numpy.linalg as la
from ctf import random as crandom
import gzip
import shutil
import os
import argparse
import arg_defs as arg_defs

sys.path.insert(0, '../SGD')
from gradient1 import sparse_SGD
sys.path.insert(0, '../ALS')
from als_sp import getALS_CG
sys.path.insert(0, '../CCD')
from ccd_sp import run_CCD

glob_comm = ctf.comm()

def getOmega(T):
    [inds, data] = T.read_local_nnz()
    data[:] = 1.
    Omega = ctf.tensor(T.shape,sp=T.sp)
    Omega.write(inds,data)
    return Omega

def read_frostt_tensor(file_name, I, J, K, use_sp_rep):
    unzipped_file_name = file_name + '.tns'
    exists = os.path.isfile(unzipped_file_name)

    if not exists:
        if glob_comm.rank() == 0:
            print('Creating ' + unzipped_file_name)
        with gzip.open(file_name + '.tns.gz', 'r') as f_in:
            with open(unzipped_file_name, 'w') as f_out:
                shutil.copyfileobj(f_in, f_out)

    T_start = ctf.tensor((I + 1, J + 1, K + 1), sp=use_sp_rep)
    if glob_comm.rank() == 0:
        print('T_start initialized')
    T_start.read_from_file(unzipped_file_name)
    if glob_comm.rank() == 0:
        print('T_start read from file')
    T = ctf.tensor((I, J, K), sp=use_sp_rep)
    if glob_comm.rank() == 0:
        print('T initialized')
    T[:, :, :] = T_start[1:, 1:, 1:]
    if glob_comm.rank() == 0:
        print('T filled by shifting from T_start')

    #T.write_to_file(unzipped_file_name)

    return T

def create_lowr_tensor(I, J, K, r, sp_frac, use_sp_rep):
    U = ctf.random.random((I, r))
    V = ctf.random.random((J, r))
    W = ctf.random.random((K, r))
    T = ctf.tensor((I, J, K), sp=use_sp_rep)
    T.fill_sp_random(1, 1, sp_frac)
    T = ctf.TTTP(T, [U, V, W])
    return T

def create_function_tensor(I, J, K, sp_frac, use_sp_rep):
    T = ctf.tensor((I, J, K), sp=use_sp_rep)
    T2 = ctf.tensor((I, J, K), sp=use_sp_rep)

    T.fill_sp_random(1, 1, sp_frac)

    sizes = [I, J, K]
    index = ["i", "j", "k"]

    for i in range(3):
        n = sizes[i]
        v = np.linspace(-1, 1, n)
        v = ctf.astensor(v ** 2)
        T2.i("ijk") << T.i("ijk") * v.i(index[i])

    [inds, data] = T2.read_local_nnz()
    data[:] **= .5
    data[:] *= -1.
    T2 = ctf.tensor(T2.shape,sp=use_sp_rep)
    T2.write(inds,data)
    
    return T2

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    arg_defs.add_general_arguments(parser)
    args, _ = parser.parse_known_args()

    I = args.I
    J = args.J
    K = args.K
    R = args.R
    
    numiter_ALS_imp = args.num_iter_ALS_implicit
    numiter_ALS_exp = args.num_iter_ALS_explicit
    numiter_CCD = args.num_iter_CCD
    numiter_SGD = args.num_iter_SGD
    objfreq_CCD = args.obj_freq_CCD
    objfreq_SGD = args.obj_freq_SGD
    time_limit = args.time_limit
    err_thresh = args.err_thresh
    sp_frac = args.sp_fraction
    use_sp_rep = args.use_sparse_rep
    block_size_ALS_imp = args.block_size_ALS_implicit
    block_size_ALS_exp = args.block_size_ALS_explicit
    reg_ALS = args.regularization_ALS
    reg_CCD = args.regularization_CCD
    reg_SGD = args.regularization_SGD
    learn_rate = args.learning_rate
    sample_frac_SGD = args.sample_frac_SGD
    use_func_tsr = args.function_tensor
    tensor_file = args.tensor_file
    

    if use_func_tsr == True:
        if ctf.comm().rank() == 0:
            print("Generating",sp_frac,"sampled function tensor")
        T = create_function_tensor(I, J, K, sp_frac, use_sp_rep)
    elif tensor_file != '':
        if ctf.comm().rank() == 0:
            print("Reading tensor from file",tensor_file)
        T = read_frostt_tensor(tensor_file, I, J, K, use_sp_rep)
    else:
        if ctf.comm().rank() == 0:
            print("Generating",sp_frac,"sampled low rank tensor")
        T = create_lowr_tensor(I, J, K, R, sp_frac, use_sp_rep)

    if T.sp:
        print("Sparse tensor shape is",T.shape,"nonzero count is",T.nnz_tot)
    else:
        print("Dense tensor shape is",T.shape)

    print("Computing tensor completion with CP rank",R)
     
    omega = getOmega(T)
    U = ctf.random.random((I, R))
    V = ctf.random.random((J, R))
    W = ctf.random.random((K, R))

    if numiter_ALS_imp > 0:
        if ctf.comm().rank() == 0:
            print("Performing up to",numiter_ALS_imp,"iterations or until reaching error threshold",err_thresh,"or reaching time limit of,",time_limit,"seconds of ALS with implicit CG")
            print("ALS with implicit CG block size is",block_size_ALS_imp,"and regularization parameter is",reg_ALS)
        U_copy = ctf.copy(U)
        V_copy = ctf.copy(V)
        W_copy = ctf.copy(W)

        getALS_CG(T,U_copy,V_copy,W_copy,reg_ALS,omega,I,J,K,R,block_size_ALS_imp,numiter_ALS_imp,err_thresh,time_limit,True) 

    if numiter_ALS_exp > 0:
        if ctf.comm().rank() == 0:
            print("Performing up to",numiter_ALS_exp,"iterations or until reaching error threshold",err_thresh,"or reaching time limit of",time_limit,"seconds of ALS with explicit CG")
            print("ALS with explicit CG block size is",block_size_ALS_exp,"and regularization parameter is",reg_ALS)
        U_copy = ctf.copy(U)
        V_copy = ctf.copy(V)
        W_copy = ctf.copy(W)

        getALS_CG(T,U_copy,V_copy,W_copy,reg_ALS,omega,I,J,K,R,block_size_ALS_exp,numiter_ALS_exp,err_thresh,time_limit,False) 


    if numiter_CCD > 0:
        if ctf.comm().rank() == 0:
            print("Performing up to",numiter_CCD,"iterations, or reaching time limit of",time_limit,"seconds of CCD")
            print("CCD regularization parameter is",reg_CCD)
        U_copy = ctf.copy(U)
        V_copy = ctf.copy(V)
        W_copy = ctf.copy(W)

        run_CCD(T,U_copy,V_copy,W_copy,omega,reg_CCD,numiter_CCD,time_limit,objfreq_CCD)

 
    if numiter_SGD > 0:
        if ctf.comm().rank() == 0:
            print("Performing up to",numiter_SGD,"iterations or until reaching error threshold",err_thresh,"or reaching time limit of",time_limit,"seconds of SGD")
            print("SGD sample fraction is",sample_frac_SGD,", learning rate is", learn_rate,"and regularization parameter is",reg_ALS)
        U_copy = ctf.copy(U)
        V_copy = ctf.copy(V)
        W_copy = ctf.copy(W)

        sparse_SGD(T, U_copy, V_copy, W_copy, reg_SGD, omega, I, J, K, R, learn_rate, sample_frac_SGD, numiter_SGD, err_thresh, time_limit, objfreq_SGD)


