import ctf, time, random
import numpy as np
from functools import reduce
import numpy.linalg as la
from ctf import random as crandom

glob_comm = ctf.comm()

INDEX_STRING = "ijklmnopq"


def stepdense_updateU(T, U, V, W, Lambda, I, J, K, r, R, stepSize):
    R = ctf.tensor((I, J, K), sp=True)
    R.i("ijk") << T.i("ijk") - U.i("iu") * V.i("ju") * W.i("ku")
    H = ctf.tensor((J, K, r))
    H.i("jku") << V.i("ju") * W.i("ku")
    prev_norm = ctf.vecnorm(R)
    for i in range(10):
        u_temp = ctf.tensor(copy=U)
        u_temp.i("ir") << - stepSize * (2 * Lambda * U.i("ir") - R.i("ijk") * H.i("jkr"))
        R.set_zero()
        R.i("ijk") << T.i("ijk") - u_temp.i("iu") * V.i("ju") * W.i("ku")
        norm = ctf.vecnorm(R)
        if norm < prev_norm:
            prev_norm = norm
            stepSize /= 2.0
        else:
            stepSize *= 2.0
    U.i("ir") << - stepSize * (2 * Lambda * U.i("ir") - R.i("ijk") * H.i("jkr"))
    return U


def dense_updateU(T, U, V, W, Lambda, I, J, K, r, R, stepSize):
    H = ctf.tensor((J, K, r))
    H.i("jku") << V.i("ju") * W.i("ku")
    U.i("ir") << - stepSize * (2 * Lambda * U.i("ir") - R.i("ijk") * H.i("jkr"))
    return U


def dense_updateV(T, U, V, W, Lambda, I, J, K, r, R, stepSize):
    H = ctf.tensor((I, K, r))
    H.i("iku") << U.i("iu") * W.i("ku")
    V.i("jr") << - stepSize * (2 * Lambda * V.i("jr") - R.i("ijk") * H.i("ikr"))
    return V


def dense_updateW(T, U, V, W, Lambda, I, J, K, r, R, stepSize):
    H = ctf.tensor((I, J, r))
    H.i("iju") << U.i("iu") * V.i("ju")
    W.i("kr") << - stepSize * (2 * Lambda * W.i("kr") - R.i("ijk") * H.i("ijr"))
    return W


def dense_update(T, factors, Lambda, sizes, rank, stepSize):
    print("used")
    dimension = len(sizes)
    indexes = INDEX_STRING[:dimension]
    tup_list = [(factors[i], indexes[i]) for i in range(dimension)]
    R = ctf.tensor(tuple(sizes))
    for i in range(dimension):
        R.i(indexes) << T.i(indexes) - reduce(lambda x, y: x[0].i(x[1] + "u") * y[0].i(y[1] + "u"), tup_list)
        H = ctf.tensor(tuple((sizes[:i] + sizes[i + 1:]).append(rank)))
        H.i(indexes[:i] + indexes[i + 1:] + "u") << reduce(lambda x, y: x[0].i(x[1] + "u") * y[0].i(y[1] + "u"),
                                                           tup_list[:i] + tup_list[i + 1:])
        factors[i].i(indexes[i] + "r") << - stepSize * (
                    2 * Lambda * factors[i].i(indexes[i] + "r") - R.i(indexes) * H.i(
                indexes[:i] + indexes[i + 1:] + "r"))
        if i < dimension - 1:
            R.set_zero()
    return ctf.vecnorm(R) + (sum([ctf.vecnorm(f) for f in factors])) * Lambda


# def sparse_updateU(T,U,V,W,Lambda,omega,I,J,K,r, stepSize):
#     H = ctf.tensor((J,K,r))
#     H.i("jku") << V.i("ju")*W.i("ku")
#     U.i("ir") << - stepSize*(2*Lambda*U.i("ir") - omega.i("ijk")*R.i("ijk")*H.i("jkr"))
#     return U
#
# def sparse_updateV(T,U,V,W,Lambda,omega,I,J,K,r,R,stepSize):
#     H = ctf.tensor((I,K,r))
#     H.i("iku") << U.i("iu")*W.i("ku")
#     V.i("jr") << - stepSize*(2*Lambda*V.i("jr") - omega.i("ijk")*R.i("ijk")*H.i("ikr"))
#     return V
#
# def sparse_updateW(T,U,V,W,Lambda,omega,I,J,K,r,R,stepSize):
#     H = ctf.tensor((I,J,r))
#     H.i("iju") << U.i("iu")*V.i("ju")
#     W.i("kr") << - stepSize*(2*Lambda*W.i("kr") - omega.i("ijk")*R.i("ijk")*H.i("ijr"))
#     return W

def sparse_updateU(T, U, V, W, Lambda, omega, I, J, K, r, R, stepSize):
    R = ctf.tensor((I, J, K), sp=True)
    R.i("ijk") << T.i("ijk") - omega.i("ijk") * U.i("iu") * V.i("ju") * W.i("ku")
    H = ctf.tensor((J, K, r))
    H.i("jku") << V.i("ju") * W.i("ku")
    U.i("ir") << - stepSize * (2 * Lambda * U.i("ir") - omega.i("ijk") * R.i("ijk") * H.i("jkr"))
    return U


def sparse_updateV(T, U, V, W, Lambda, omega, I, J, K, r, R, stepSize):
    H = ctf.tensor((I, K, r))
    H.i("iku") << U.i("iu") * W.i("ku")
    V.i("jr") << - stepSize * (2 * Lambda * V.i("jr") - omega.i("ijk") * R.i("ijk") * H.i("ikr"))
    return V


def sparse_updateW(T, U, V, W, Lambda, omega, I, J, K, r, R, stepSize):
    H = ctf.tensor((I, J, r))
    H.i("iju") << U.i("iu") * V.i("ju")
    W.i("kr") << - stepSize * (2 * Lambda * W.i("kr") - omega.i("ijk") * R.i("ijk") * H.i("ijr"))
    return W


def sparse_update(T, factors, Lambda, sizes, rank, stepSize):
    omega = getOmega(T)
    dimension = len(sizes)
    indexes = INDEX_STRING[:dimension]
    R = ctf.tensor(tuple(sizes))

    for i in range(dimension):
        tup_list = [factors[i].i(indexes[i] + "r") for i in range(dimension)]
        #R.i(indexes) << T.i(indexes) - omega.i(indexes) * reduce(lambda x, y: x * y, tup_list)
        R.i(indexes) << T.i(indexes) - ctf.TTTP(omega, factors).i(indexes)
        H = ctf.tensor(tuple((sizes[:i] + sizes[i + 1:] + [rank])))
        H.i(indexes[:i] + indexes[i + 1:] + "r") << reduce(lambda x, y: x * y, tup_list[:i] + tup_list[i + 1:])
        factors[i].i(indexes[i] + "r") << - stepSize * (
                    2 * Lambda * factors[i].i(indexes[i] + "r") - H.i(indexes[:i] + indexes[i + 1:] + "r")) * omega.i(indexes) * R.i(indexes)
        if i < dimension - 1:
            R.set_zero()
    #return ctf.vecnorm(R) + (sum([ctf.vecnorm(f) for f in factors])) * Lambda


def sparse_update_steps(T, factors, Lambda, sizes, rank, stepSize):
    omega = getOmega(T)
    dimension = len(sizes)
    indexes = INDEX_STRING[:dimension]
    R = ctf.tensor(tuple(sizes), sp=True)
    Steps = []
    for i in range(dimension):
        tup_list = [factors[i].i(indexes[i] + "r") for i in range(dimension)]
        R.i(indexes) << T.i(indexes) - reduce(lambda x, y: x * y, tup_list) * omega.i(indexes)
        H = ctf.tensor(tuple((sizes[:i] + sizes[i + 1:] + [rank])))
        H.i(indexes[:i] + indexes[i + 1:] + "r") << reduce(lambda x, y: x * y, tup_list[:i] + tup_list[i + 1:])

        this_step = stepSize
        gradient = ctf.tensor((sizes[i], rank))
        gradient.i(indexes[i] + "r") << 2 * Lambda * factors[i].i(indexes[i] + "r") - omega.i(indexes) * R.i(
            indexes) * H.i(indexes[:i] + indexes[i + 1:] + "r")
        stop = False
        temp_f = ctf.tensor(copy=factors[i])
        temp_f.i(indexes[i] + "r") << - this_step * gradient.i(indexes[i] + "r")

        current_norm = float("inf")
        while not stop:
            temp_f.i(indexes[i] + "r") << - this_step * gradient.i(indexes[i] + "r")
            tup_list[i] = temp_f.i(indexes[i] + "r")
            R.set_zero()
            R.i(indexes) << T.i(indexes) - omega.i(indexes) * reduce(lambda x, y: x * y, tup_list)
            this_norm = ctf.vecnorm(R) + (
                sum([ctf.vecnorm(f) for f in factors[:i] + factors[i + 1:] + [temp_f]])) * Lambda
            if this_norm < current_norm:
                current_norm = this_norm
                this_step *= 2.0
            else:
                Steps.append(this_step)
                break
        factors[i].i(indexes[i] + "r") << - this_step * gradient.i(indexes[i] + "r")
        R.set_zero()
    tup_list = [factors[i].i(indexes[i] + "r") for i in range(dimension)]
    R.i(indexes) << T.i(indexes) - omega.i(indexes) * reduce(lambda x, y: x * y, tup_list)
    return ctf.vecnorm(R) + (sum([ctf.vecnorm(f) for f in factors])) * Lambda, Steps


def dense_GD(T, U, V, W, Lambda, I, J, K, r, stepSize):
    iteration_count = 0
    E = ctf.tensor((I, J, K))
    E.i("ijk") << T.i("ijk") - U.i("iu") * V.i("ju") * W.i("ku")
    curr_err_norm = ctf.vecnorm(E) + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W)) * Lambda

    while True:
        next_err_norm = dense_update(T, [U, V, W], Lambda, [I, J, K], r, stepSize)
        # U = dense_updateU(T,U,V,W,Lambda,I,J,K,r,E,stepSize)
        # V = dense_updateV(T,U,V,W,Lambda,I,J,K,r,E,stepSize)
        # W = dense_updateW(T,U,V,W,Lambda,I,J,K,r,E,stepSize)
        # E.set_zero()
        # E.i("ijk") << T.i("ijk") - U.i("iu")*V.i("ju")*W.i("ku")
        # next_err_norm = ctf.vecnorm(E) + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W))*Lambda
        if abs(curr_err_norm - next_err_norm) < .001 or iteration_count > 100:
            break

        print(curr_err_norm, next_err_norm)
        curr_err_norm = next_err_norm
        iteration_count += 1
    print("Number of iterations: ", iteration_count)
    return U, V, W


def sparse_GD(T, U, V, W, Lambda, omega, I, J, K, r, stepSize):
    iteration_count = 0
    R = ctf.tensor((I, J, K), sp=True)
    R.i("ijk") << T.i("ijk") - omega.i("ijk") * U.i("iu") * V.i("ju") * W.i("ku")
    curr_err_norm = ctf.vecnorm(R) + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W)) * Lambda
    norm = [curr_err_norm]
    while True:
        next_err_norm = sparse_update(T, [U, V, W], Lambda, [I, J, K], r, stepSize)
        if abs(curr_err_norm - next_err_norm) < .0001 or iteration_count > 100:
            break

        print(curr_err_norm, next_err_norm)
        curr_err_norm = next_err_norm
        norm.append(curr_err_norm)
        iteration_count += 1

    print("Number of iterations: ", iteration_count)
    return norm


def test_sparse_GD(T, U, V, W, Lambda, omega, I, J, K, r, stepSize):
    iteration_count = 0
    R = ctf.tensor((I, J, K), sp=True)
    R.i("ijk") << T.i("ijk") - omega.i("ijk") * U.i("iu") * V.i("ju") * W.i("ku")
    curr_err_norm = ctf.vecnorm(R) + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W)) * Lambda
    norm = [curr_err_norm]
    stepSizes = [[] for i in range(3)]
    while True:
        next_err_norm, steps = sparse_update_steps(T, [U, V, W], Lambda, [I, J, K], r, stepSize)
        for i in range(len(steps)):
            stepSizes[i].append(steps[i])
        if abs(curr_err_norm - next_err_norm) < .001 or iteration_count > 50:
            break

        print(curr_err_norm, next_err_norm)
        curr_err_norm = next_err_norm
        norm.append(curr_err_norm)
        iteration_count += 1

    print("Number of iterations: ", iteration_count)
    return norm, stepSizes


def sparse_SGD(T, U, V, W, Lambda, omega, I, J, K, r, stepSize, sample_rate):
    iteration_count = 0
    R = ctf.tensor((I, J, K), sp=True)
    R.i("ijk") << T.i("ijk") - ctf.TTTP(omega, [U, V, W]).i("ijk")
    # R.i("ijk") << T.i("ijk") - U.i("iu") * V.i("ju") * W.i("ku") * omega.i("ijk")
    curr_err_norm = ctf.vecnorm(R) + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W)) * Lambda
    norm = [curr_err_norm]
    work_cycle = int(1.0 / sample_rate)
    step = stepSize * 0.5
    starting_time = time.time()
    while True:
        iteration_count += 1
        sampled_T = T.copy()
        sampled_T.sample(sample_rate)

        sparse_update(sampled_T, [U, V, W], Lambda, [I, J, K], r, stepSize * 0.5 + step)
        step *= 0.99
        sampled_T.set_zero()

        if iteration_count % work_cycle == 0:
            R.set_zero()
            #R.i("ijk") << T.i("ijk") - U.i("iu") * V.i("ju") * W.i("ku") * omega.i("ijk")
            R.i("ijk") << T.i("ijk") - ctf.TTTP(omega, [U, V, W]).i("ijk")
            diff_norm = ctf.vecnorm(R)
            next_err_norm = diff_norm + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W)) * Lambda
            #print(curr_err_norm, next_err_norm, diff_norm)
            if ctf.comm().rank() == 0:
                print(diff_norm, time.time() - starting_time)

            if abs(curr_err_norm - next_err_norm) < .001 or iteration_count > work_cycle * 50:
                break

            curr_err_norm = next_err_norm
            norm.append(curr_err_norm)
    if ctf.comm().rank() == 0:
        print("Number of iterations: ", iteration_count)
    return norm


def test_sparse_SGD(T, U, V, W, Lambda, omega, I, J, K, r, stepSize, sample_rate):
    iteration_count = 0
    R = ctf.tensor((I, J, K), sp=True)
    R.i("ijk") << T.i("ijk") - U.i("iu") * V.i("ju") * W.i("ku") * omega.i("ijk")
    curr_err_norm = ctf.vecnorm(R) + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W)) * Lambda
    norm = [curr_err_norm]
    stepSizes = [[] for i in range(3)]

    work_cycle = int(1.0 / sample_rate)
    print("work_cycle", work_cycle)
    while True:
        sampled_T = T.copy()
        sampled_T.sample(sample_rate)
        next_err_norm, steps = sparse_update_steps(sampled_T, [U, V, W], Lambda, [I, J, K], r, stepSize)
        for i in range(len(steps)):
            stepSizes[i].append(steps[i])
        sampled_T.set_zero()

        if iteration_count % work_cycle == 0:
            R.set_zero()
            R.i("ijk") << T.i("ijk") - U.i("iu") * V.i("ju") * W.i("ku") * omega.i("ijk")
            diff_norm = ctf.vecnorm(R)
            next_err_norm = diff_norm + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W)) * Lambda
            print(curr_err_norm, next_err_norm, diff_norm)

            if abs(curr_err_norm - next_err_norm) < .0001 or iteration_count > work_cycle * 100:
                break

            curr_err_norm = next_err_norm
            norm.append(curr_err_norm)
        iteration_count += 1

    #print("Number of iterations: ", iteration_count)
    return norm, stepSizes



def dense_SGD(T, U, V, W, Lambda, I, J, K, r, stepSize, sample_rate):
    iteration_count = 0
    E = ctf.tensor((I, J, K))
    R = ctf.tensor((I, J, K))
    R.i("ijk") << T.i("ijk") - U.i("iu") * V.i("ju") * W.i("ku")
    sampled_T = ctf.tensor((I, J, K))
    curr_err_norm = ctf.vecnorm(R) + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W)) * Lambda
    while True:
        random = ctf.tensor((I, J, K))
        random.fill_random(0, 1)
        random = ((random < sample_rate) * ctf.astensor(1.))
        sampled_T = T * random
        # sampled_T.sample(sample_rate)
        sampled_omega = getOmega(sampled_T)
        E.i("ijk") << sampled_T.i("ijk") - sampled_omega.i("ijk") * U.i("iu") * V.i("ju") * W.i("ku")
        U = sparse_updateU(sampled_T, U, V, W, Lambda, sampled_omega, I, J, K, r, E, stepSize)
        E.i("ijk") << sampled_T.i("ijk") - sampled_omega.i("ijk") * U.i("iu") * V.i("ju") * W.i("ku")
        V = sparse_updateV(sampled_T, U, V, W, Lambda, sampled_omega, I, J, K, r, E, stepSize)
        E.i("ijk") << sampled_T.i("ijk") - sampled_omega.i("ijk") * U.i("iu") * V.i("ju") * W.i("ku")
        W = sparse_updateW(sampled_T, U, V, W, Lambda, sampled_omega, I, J, K, r, E, stepSize)
        E.set_zero()
        sampled_T.set_zero()

        if iteration_count % 5 == 0:
            R.set_zero()
            R.i("ijk") << T.i("ijk") - U.i("iu") * V.i("ju") * W.i("ku")
            next_err_norm = ctf.vecnorm(R) + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W)) * Lambda
            print(curr_err_norm, next_err_norm)

            if abs(curr_err_norm - next_err_norm) < .001 or iteration_count > 10:
                break

            curr_err_norm = next_err_norm

        iteration_count += 1

    print("Number of iterations: ", iteration_count)
    return U, V, W


def function_tensor(I, J, K, sparsity):
    # N = 5
    # n = 51
    # L = 100
    # nsample = 10*N*n*L #10nNL = 255000

    T = ctf.tensor((I, J, K), sp=True)
    T2 = ctf.tensor((I, J, K), sp=True)

    T.fill_sp_random(1, 1, sparsity)
    # T = ctf.exp(-1 * ctf.power(ctf.power(T,2),0.5))  # x = exp(-sqrt(x^2))

    sizes = [I, J, K]
    index = ["i", "j", "k"]

    for i in range(3):
        n = sizes[i]
        v = np.linspace(-1, 1, n)
        # v = np.arange(1,n+1)
        v = ctf.astensor(v ** 2)

        v2 = ctf.tensor(n, sp=True)
        v2 = v
        T2.i("ijk") << T.i("ijk") * v2.i(index[i])

    T2 = ctf.power(T2, 0.5)
    T2 = (-1.0) * T2

    # T2 = ctf.exp(T2)

    return T2

def getOmega(T):
    [inds, data] = T.read_local_nnz()
    data[:] = 1.
    Omega = ctf.tensor(T.shape,sp=True)
    Omega.write(inds,data)
    return Omega



def main():
    s = 10
    sparsity = 10
    r = 5
    regParam = 0.00001
    stepSize = 0.01
    sample_rate = 0.05
    w = ctf.comm()
    I = s
    J = s
    K = s
    T = ctf.tensor((I, J, K), sp=True)
    T.fill_sp_random(0., 1., 1. / sparsity)
    # T.read_from_file("T.txt")
    Omega = getOmega(T)
    s1 = T.sum()
    s2 = Omega.sum()
    if ctf.comm().rank() == 0:
        print(s1,s2)
    #T = function_tensor(I, J, K, sparsity)
    U = ctf.random.random((I, r))
    V = ctf.random.random((J, r))
    W = ctf.random.random((K, r))

    #T.write_to_file("T.txt")
    sparse_SGD(T, ctf.tensor(copy=U), ctf.tensor(copy=V), ctf.tensor(copy=W), regParam, Omega, I, J, K, r, stepSize, sample_rate)

main()
