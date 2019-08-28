import ctf, time, random, sys
import numpy as np
from functools import reduce
import numpy.linalg as la
from ctf import random as crandom

glob_comm = ctf.comm()
import gzip
import shutil
import os

INDEX_STRING = "ijklmnopq"

def sparse_update(T, factors, Lambda, sizes, rank, stepSize, sample_rate, times):
    starting_time = time.time()
    t_go = ctf.timer("SGD_getOmega")
    t_go.start()
    omega = getOmega(T)
    t_go.stop()
    dimension = len(sizes)
    indexes = INDEX_STRING[:dimension]
    R = ctf.tensor(copy=T) #ctf.tensor(tuple(sizes), sp = True)
    times[2] += time.time() - starting_time
    for i in range(dimension):
        tup_list = [factors[i].i(indexes[i] + "r") for i in range(dimension)]
        #R.i(indexes) << T.i(indexes) - omega.i(indexes) * reduce(lambda x, y: x * y, tup_list)
        starting_time = time.time()
        R.i(indexes) << -1.* ctf.TTTP(omega, factors).i(indexes)
        times[3] += time.time() - starting_time
        starting_time = time.time()
        #H = ctf.tensor(tuple((sizes[:i] + sizes[i + 1:] + [rank])))
        times[4] += time.time() - starting_time
        starting_time = time.time()
        #H.i(indexes[:i] + indexes[i + 1:] + "r") << 
        Hterm = reduce(lambda x, y: x * y, tup_list[:i] + tup_list[i + 1:])
        times[5] += time.time() - starting_time
        starting_time = time.time()
        t_ctr = ctf.timer("SGD_main_contraction")
        t_ctr.start()
        (1- stepSize * 2 * Lambda * sample_rate)*factors[i].i(indexes[i] + "r") << stepSize * Hterm * R.i(indexes)
        t_ctr.stop()
        times[6] += time.time() - starting_time
        if i < dimension - 1:
            R = ctf.tensor(copy=T)
    #return ctf.vecnorm(R) + (sum([ctf.vecnorm(f) for f in factors])) * Lambda

def sparse_SGD(T, U, V, W, Lambda, omega, I, J, K, r, stepSize, sample_rate, num_iter, errThresh, time_limit, work_cycle):
    times = [0 for i in range(7)]

    iteration_count = 0
    total_count = 0
    R = ctf.tensor((I, J, K), sp=T.sp)
    if T.sp == True:
        nnz_tot = T.nnz_tot
    else:
        nnz_tot = ctf.sum(omega)
    start_time = time.time()
    starting_time = time.time()
    dtime = 0
    R.i("ijk") << T.i("ijk") - ctf.TTTP(omega, [U, V, W]).i("ijk")
    # R.i("ijk") << T.i("ijk") - U.i("iu") * V.i("ju") * W.i("ku") * omega.i("ijk")
    curr_err_norm = ctf.vecnorm(R) + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W)) * Lambda
    times[0] += time.time() - starting_time
    norm = [curr_err_norm]
    #work_cycle = 1 #int(1.0 / sample_rate)
    step = stepSize * 0.5
    t_obj_calc = 0.

    while iteration_count < num_iter and time.time() - start_time - t_obj_calc < time_limit:
        iteration_count += 1
        starting_time = time.time()
        sampled_T = T.copy()
        sampled_T.sample(sample_rate)
        times[1] += time.time() - starting_time

        sparse_update(sampled_T, [U, V, W], Lambda, [I, J, K], r, stepSize * 0.5 + step, sample_rate, times)
        step *= 0.99
        sampled_T.set_zero()

        if iteration_count % work_cycle == 0:
            duration = time.time() - start_time - t_obj_calc
            t_b_obj = time.time()
            total_count += 1
            R.set_zero()
            #R.i("ijk") << T.i("ijk") - U.i("iu") * V.i("ju") * W.i("ku") * omega.i("ijk")
            R.i("ijk") << T.i("ijk") - ctf.TTTP(omega, [U, V, W]).i("ijk")
            diff_norm = ctf.vecnorm(R)
            RMSE = diff_norm/(nnz_tot**.5)
            next_err_norm = diff_norm + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W)) * Lambda
            if glob_comm.rank() == 0:
                print('Objective after',duration,'seconds (',iteration_count,'iterations) is: {}'.format(next_err_norm))
                print('RMSE after',duration,'seconds (',iteration_count,'iterations) is: {}'.format(RMSE))
            t_obj_calc += time.time() - t_b_obj
            #print(curr_err_norm, next_err_norm, diff_norm)
            #if ctf.comm().rank() == 0:
            #    x = time.time() - start_time
            #    print(diff_norm, x, total_count, x/total_count)
            #    # print(times)

            if abs(curr_err_norm - next_err_norm) < errThresh:
                break

            curr_err_norm = next_err_norm
            norm.append(curr_err_norm)

    duration = time.time() - start_time - t_obj_calc
    if ctf.comm().rank() == 0:
        print('SGD amortized seconds per sweep: {}'.format(duration/(iteration_count*sample_rate)))
        print("Time/SGD iteration: {}".format(duration/iteration_count))
    #curr_err_norm = ctf.vecnorm(R) + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W)) * Lambda
    #R.i("ijk") << T.i("ijk") - ctf.TTTP(omega, [U, V, W]).i("ijk")
    #if ctf.comm().rank() == 0:
    #    print('Objective after',duration,'seconds is: {}'.format(curr_err_norm))
    return norm

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
    regParam = 0.00001

    file_name = sys.argv[1]
    I = int(sys.argv[2])
    J = int(sys.argv[3])
    K = int(sys.argv[4])
    stepSize = int(sys.argv[5])
    r = int(sys.argv[6])
    sample_rate = float(sys.argv[7])

    T = read_from_frostt(file_name, I, J, K)
    # T.read_from_file("T.txt")
    Omega = getOmega(T)

    s = T.sum()
    os = Omega.sum()
    if ctf.comm().rank() == 0:
        print(s, os)
        print (sys.argv)
    #T = function_tensor(I, J, K, sparsity)
    U = ctf.random.random((I, r))
    V = ctf.random.random((J, r))
    W = ctf.random.random((K, r))

    #T.write_to_file("T.txt")
    sparse_SGD(T, U, V, W, regParam, Omega, I, J, K, r, stepSize, sample_rate, 100, 1.e-5, 1.E3, 30, 10)

#main()
if __name__ == '__main__':
    main()

