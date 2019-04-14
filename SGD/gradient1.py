import ctf, time, random
import numpy as np
from functools import reduce
import numpy.linalg as la
from ctf import random as crandom

glob_comm = ctf.comm()

INDEX_STRING = "ijklmnopq"

def sparse_update(T, factors, Lambda, sizes, rank, stepSize, sample_rate, times):
    starting_time = time.time()
    omega = getOmega(T)
    dimension = len(sizes)
    indexes = INDEX_STRING[:dimension]
    R = ctf.tensor(tuple(sizes), sp = True)
    times[2] += time.time() - starting_time
    for i in range(dimension):
        tup_list = [factors[i].i(indexes[i] + "r") for i in range(dimension)]
        #R.i(indexes) << T.i(indexes) - omega.i(indexes) * reduce(lambda x, y: x * y, tup_list)
        starting_time = time.time()
        R.i(indexes) << T.i(indexes) - ctf.TTTP(omega, factors).i(indexes)
        times[3] += time.time() - starting_time
        starting_time = time.time()
        if ctf.comm().rank() == 0:
            print("check")
        H = ctf.tensor(tuple((sizes[:i] + sizes[i + 1:] + [rank])))
        times[4] += time.time() - starting_time
        starting_time = time.time()
        H.i(indexes[:i] + indexes[i + 1:] + "r") << reduce(lambda x, y: x * y, tup_list[:i] + tup_list[i + 1:])
        times[5] += time.time() - starting_time
        starting_time = time.time()
        factors[i].i(indexes[i] + "r") << - stepSize * (2 * Lambda * sample_rate * factors[i].i(indexes[i] + "r") - H.i(indexes[:i] + indexes[i + 1:] + "r") * R.i(indexes))
        times[6] += time.time() - starting_time
        if i < dimension - 1:
            R.set_zero()
    #return ctf.vecnorm(R) + (sum([ctf.vecnorm(f) for f in factors])) * Lambda

def sparse_SGD(T, U, V, W, Lambda, omega, I, J, K, r, stepSize, sample_rate):
    times = [0 for i in range(7)]

    iteration_count = 0
    total_count = 0
    R = ctf.tensor((I, J, K), sp=True)
    start_time = time.time()
    starting_time = time.time()
    R.i("ijk") << T.i("ijk") - ctf.TTTP(omega, [U, V, W]).i("ijk")
    # R.i("ijk") << T.i("ijk") - U.i("iu") * V.i("ju") * W.i("ku") * omega.i("ijk")
    curr_err_norm = ctf.vecnorm(R) + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W)) * Lambda
    times[0] += time.time() - starting_time
    norm = [curr_err_norm]
    work_cycle = int(1.0 / sample_rate)
    step = stepSize * 0.5

    while True:
        iteration_count += 1
        starting_time = time.time()
        sampled_T = T.copy()
        sampled_T.sample(sample_rate)
        times[1] += time.time() - starting_time

        sparse_update(sampled_T, [U, V, W], Lambda, [I, J, K], r, stepSize * 0.5 + step, sample_rate, times)
        step *= 0.99
        sampled_T.set_zero()

        if iteration_count % work_cycle == 0:
            total_count += 1
            R.set_zero()
            #R.i("ijk") << T.i("ijk") - U.i("iu") * V.i("ju") * W.i("ku") * omega.i("ijk")
            R.i("ijk") << T.i("ijk") - ctf.TTTP(omega, [U, V, W]).i("ijk")
            diff_norm = ctf.vecnorm(R)
            next_err_norm = diff_norm + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W)) * Lambda
            #print(curr_err_norm, next_err_norm, diff_norm)
            if ctf.comm().rank() == 0:
                print(diff_norm, time.time() - start_time, total_count)
                print(times)

            if abs(curr_err_norm - next_err_norm) < .001 or iteration_count > work_cycle * 50:
                break

            curr_err_norm = next_err_norm
            norm.append(curr_err_norm)
    if ctf.comm().rank() == 0:
        print("Number of iterations: ", iteration_count)
    return norm

def getOmega(T):
    [inds, data] = T.read_local_nnz()
    data[:] = 1.
    Omega = ctf.tensor(T.shape,sp=True)
    Omega.write(inds,data)
    return Omega

def main():
    s = 2000
    sparsity = 100
    r = 16
    regParam = 0.00001
    stepSize = 0.01
    sample_rate = 0.01

    I = s
    J = s
    K = s
    T = ctf.tensor((I, J, K), sp=True)
    T.fill_sp_random(0., 1., 1. / sparsity)
    # T.read_from_file("T.txt")
    Omega = getOmega(T)

    s = T.sum()
    os = Omega.sum()
    if ctf.comm().rank() == 0:
        print(s, os)
    #T = function_tensor(I, J, K, sparsity)
    U = ctf.random.random((I, r))
    V = ctf.random.random((J, r))
    W = ctf.random.random((K, r))

    #T.write_to_file("T.txt")
    sparse_SGD(T, ctf.tensor(copy=U), ctf.tensor(copy=V), ctf.tensor(copy=W), regParam, Omega, I, J, K, r, stepSize, sample_rate)

main()

