import ctf, time, random, sys
import numpy as np
from functools import reduce
import numpy.linalg as la
from ctf import random as crandom
import gzip
import shutil
import os

sys.path.append('/../SGD')
from gradient1.py import *

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
    #generate tensor
    file_name = sys.argv[1]
    I = int(sys.argv[2])
    J = int(sys.argv[3])
    K = int(sys.argv[4])
    T = read_from_frostt(file_name, I, J, K)
    # T.read_from_file("T.txt")
    Omega = getOmega(T)
    s = T.sum()
    os = Omega.sum()
    if ctf.comm().rank() == 0:
        print(s, os)
        print(sys.argv)
    r = int(sys.argv[5])
    U = ctf.random.random((I, r))
    V = ctf.random.random((J, r))
    W = ctf.random.random((K, r))

    #SGD CODE
    stepSize = int(sys.argv[6])
    sample_rate = float(sys.argv[7])
    regParam = 0.00001
    if ctf.comm().rank() == 0:
        print("Begin SGD")
    sparse_SGD(T, U, V, W, regParam, Omega, I, J, K, r, stepSize, sample_rate)
    if ctf.comm().rank() == 0:
        print("End SGD")

    #ALS CODE

    #CCD CODE

main()
