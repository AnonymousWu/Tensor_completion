import ctf
from ctf import random
import matplotlib.pyplot as plt
import time
import sys
glob_comm = ctf.comm()
import numpy as np
status_prints = True

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

    print(1)
    T2 = ctf.power(T2, 0.5)
    print(2)
    T2 = (-1.0) * T2

    # T2 = ctf.exp(T2)

    return T2

def getOmegaOld(T):
    if not T.sp:
        omegactf = ((T != 0)*ctf.astensor(1.))
    else:
        omegactf = T / T
        assert(omegactf.sp)
    
    return omegactf

def getOmega(T):
    t_om = ctf.timer("ccd_getOmega")
    t_om.start()
    [inds, data] = T.read_local_nnz()
    data[:] = 1.
    Omega = ctf.tensor(T.shape,sp=True)
    Omega.write(inds,data)
    t_om.stop()
    return Omega

def get_objective(T,U,V,W,I,J,K,omega,regParam):
    t_obj = ctf.timer("ccd_get_objective")
    t_obj.start()
    L = ctf.tensor((I,J,K), sp=True)
    t0 = time.time()
    L.i("ijk") << T.i("ijk") - ctf.TTTP(omega, [U,V,W]).i("ijk")
    assert(L.sp)
    t1 = time.time()
    objective = ctf.vecnorm(L) + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W)) * regParam
    t2 = time.time()
    if glob_comm.rank() == 0 and status_prints == True:
        print('generate L takes {}'.format(t1 - t0))
        print('calc objective takes {}'.format(t2 - t1))
    t_obj.stop()
    return objective

def main():

    sparsity = .1
    r = 10
    num_iter = 1
    objective_frequency = 2

    if len(sys.argv) >= 4:
        I = int(sys.argv[1])
        J = int(sys.argv[2])
        K = int(sys.argv[3])
    if len(sys.argv) >= 5:
        sparsity = np.float64(sys.argv[4])
    if len(sys.argv) >= 6:
        r = int(sys.argv[5])
    if len(sys.argv) >= 7:
        num_iter = int(sys.argv[6])
    if len(sys.argv) >= 8:
    	objective_frequency = int(sys.argv[7])

    if glob_comm.rank() == 0:
        print("I is",I,"J is",J,"K is",K,"sparisty is",sparsity,"r is",r,"num_iter is",num_iter)

    # sparsity = .001
    regParam = .1

    ctf.random.seed(42)

    # # 3rd-order tensor
    T = ctf.tensor((I,J,K), sp=True)
    # # T.read_from_file('tensor.txt')
    T.fill_sp_random(0,1,sparsity)
    # T = function_tensor(I,J,K,sparsity)
    assert(T.sp)

    t0 = time.time()
    omega = getOmega(T)
    assert(omega.sp)
    # print(T.sum(), omega.sum())

    
    if glob_comm.rank() == 0:
        print('getOmega takes {}'.format(time.time() - t0))


    U = ctf.random.random((I, r))
    V = ctf.random.random((J, r))
    W = ctf.random.random((K, r))
    U_vec_list = []
    V_vec_list = []
    W_vec_list = []
    for f in range(r):
    	U_vec_list.append(U[:,f])
    	V_vec_list.append(V[:,f])
    	W_vec_list.append(W[:,f])


    # print(T)
    # T.write_to_file('tensor_out.txt')
    # assert(T.sp == 1)

    ite = 0
    objectives = []

    t_before_loop = time.time()

    t_CCD = ctf.timer_epoch("ccd_CCD")
    t_CCD.begin()
    while True:

        t_iR_upd = ctf.timer("ccd_init_R_upd")
        t_iR_upd.start()
        t0 = time.time()
        R = ctf.copy(T)
        assert(R.sp)
        t1 = time.time()
        # R -= ctf.einsum('ijk, ir, jr, kr -> ijk', omega, U, V, W)
        R -= ctf.TTTP(omega, [U,V,W])
        assert(R.sp)
        t2 = time.time()
        # R += ctf.einsum('ijk, i, j, k -> ijk', omega, U[:,0], V[:,0], W[:,0])
        R += ctf.TTTP(omega, [U[:,0], V[:,0], W[:,0]])
        assert(R.sp)
        t3 = time.time()

        t_iR_upd.stop()

        if status_prints == True and ite % objective_frequency == 0:
        	objective = get_objective(T,U,V,W,I,J,K,omega,regParam)
        	objectives.append(objective)
        	if glob_comm.rank() == 0:
        			print('Objective: {}'.format(objective))


        if glob_comm.rank() == 0 and status_prints == True:
            print('ctf.copy() takes {}'.format(t1-t0))
            print('ctf.TTTP() takes {}'.format(t2-t1))
            print('ctf.TTTP() takes {}'.format(t3-t2))


        for f in range(r):
            
            # update U[:,f]
            if glob_comm.rank() == 0 and status_prints == True:
                print('updating U[:,{}]'.format(f))

            t0 = time.time()
            # alphas = ctf.einsum('ijk, j, k -> i', R, V[:,f], W[:,f])
            alphas = ctf.einsum('ijk, j, k -> i', R, V_vec_list[f], W_vec_list[f])
            
            t1 = time.time()

            # betas = ctf.einsum('ijk, j, j, k, k -> i', omega, V[:,f], V[:,f], W[:,f], W[:,f])
            betas = ctf.einsum('ijk, j, j, k, k -> i', omega, V_vec_list[f], V_vec_list[f], W_vec_list[f], W_vec_list[f])
            # betas = ctf.tensor(I,)
            # betas.i("i") << V[:,f].i("j")*W[:,f].i("k")*ctf.TTTP(omega, [None,V[:,f],W[:,f]]).i("ijk")
            
            t2 = time.time()
            
            # U[:,f] = alphas / (regParam + betas)
            U_vec_list[f] = alphas / (regParam + betas)
            U[:,f] = U_vec_list[f]

            if glob_comm.rank() == 0 and status_prints == True:
                print('ctf.einsum() takes {}'.format(t1-t0))
                print('ctf.einsum() takes {}'.format(t2-t1))


            # update V[:,f]
            if glob_comm.rank() == 0 and status_prints == True:
                print('updating V[:,{}]'.format(f))
            # alphas = ctf.einsum('ijk, i, k -> j', R, U[:,f], W[:,f])
            alphas = ctf.einsum('ijk, i, k -> j', R, U_vec_list[f], W_vec_list[f])

            # betas = ctf.einsum('ijk, i, i, k, k -> j', omega, U[:,f], U[:,f], W[:,f], W[:,f])
            betas = ctf.einsum('ijk, i, i, k, k -> j', omega, U_vec_list[f], U_vec_list[f], W_vec_list[f], W_vec_list[f])
            
            # V[:,f] = alphas / (regParam + betas)
            V_vec_list[f] = alphas / (regParam + betas)
            V[:,f] = V_vec_list[f]


            # update W[:,f]
            if glob_comm.rank() == 0 and status_prints == True:
                print('updating W[:,{}]'.format(f))
            # alphas = ctf.einsum('ijk, i, j -> k', R, U[:,f], V[:,f])
            alphas = ctf.einsum('ijk, i, j -> k', R, U_vec_list[f], V_vec_list[f])

            # betas = ctf.einsum('ijk, i, i, j, j -> k', omega, U[:,f], U[:,f], V[:,f], V[:,f])
            betas = ctf.einsum('ijk, i, i, j, j -> k', omega, U_vec_list[f], U_vec_list[f], V_vec_list[f], V_vec_list[f])
            
            # W[:,f] = alphas / (regParam + betas)
            W_vec_list[f] = alphas / (regParam + betas)
            W[:,f] = W_vec_list[f]



            # t0 = time.time()
            # R -= ctf.einsum('ijk, i, j, k -> ijk', omega, U[:,f], V[:,f], W[:,f])
            t_tttp = ctf.timer("ccd_TTTP")
            t_tttp.start()
            # R -= ctf.TTTP(omega, [U[:,f], V[:,f], W[:,f]])
            R -= ctf.TTTP(omega, [U_vec_list[f], V_vec_list[f], W_vec_list[f]])

            # R += ctf.einsum('ijk, i, j, k -> ijk', omega, U[:,f+1], V[:,f+1], W[:,f+1])
            # R += ctf.TTTP(omega, [U[:,f+1], V[:,f+1], W[:,f+1]])
            if f+1 < r:
            	R += ctf.TTTP(omega, [U_vec_list[f+1], V_vec_list[f+1], W_vec_list[f+1]])

            t_tttp.stop()
            assert(R.sp)
            # print(time.time() - t0)

            # print(R)
            # exit(0)
        t_iR_upd.stop()
        
        ite += 1

        if ite == num_iter:
            break

    t_CCD.end()
    objective = get_objective(T,U,V,W,I,J,K,omega,regParam)

    if glob_comm.rank() == 0:
        print('Time/Iteration: {}'.format((time.time() - t_before_loop)/num_iter))
        print('Objective: {}'.format(objective))

    # plt.plot(objectives)
    # plt.yscale('log')
    # plt.show()
    # print(len(objectives))

if __name__ == '__main__':
    main()
