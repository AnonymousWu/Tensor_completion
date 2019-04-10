import ctf,time,random
import numpy as np
import numpy.linalg as la
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
from ctf import random as crandom
glob_comm = ctf.comm()
#from scipy.sparse.linalg import lsqr as lsqr
import sys
sys.path.append('../data/function_tensor')
#from data.function_tensor.function_tensor import *
from function_tensor import *
from math import sqrt

status_prints = False
CG_thresh = 1.e-10
sparse_format = True

class implicit_ATA:
    def __init__(self, f1, f2, omega, string):
        self.f1 = f1
        self.f2 = f2
        self.omega = omega
        self.string = string

    def mul(self, idx, sk):
        if self.string=="U":
            return  self.f1.i("J"+idx[1]) \
                   *self.f2.i("K"+idx[1]) \
                   *ctf.TTTP(self.omega, [sk, self.f1, self.f2]).i(idx[0]+"JK")
        if self.string=="V":
            return  self.f1.i("I"+idx[1]) \
                   *self.f2.i("K"+idx[1]) \
                   *ctf.TTTP(self.omega, [self.f1, sk, self.f2]).i("I"+idx[0]+"K")
        if self.string=="W":
            return  self.f1.i("I"+idx[1]) \
                   *self.f2.i("J"+idx[1]) \
                   *ctf.TTTP(self.omega, [self.f1, self.f2, sk]).i("IJ"+idx[0])

def CG(A,b,x0,r,regParam,I,is_implicit=False):

    t_batch_cg = ctf.timer("ALS_exp_cg")
    t_batch_cg.start()
    
    Ax0 = ctf.tensor((I,r))
    if is_implicit:
        Ax0.i("ir") << A.mul("ir",x0)
    else:
        Ax0.i("ir") << A.i("irl")*x0.i("il")
    Ax0 += regParam*x0 
    rk = b - Ax0
    sk = rk
    xk = x0
    for i in range(sk.shape[-1]): # how many iterations?
        Ask = ctf.tensor((I,r))
        t_cg_bmvec = ctf.timer("ALS_exp_cg_mvec")
        t_cg_bmvec.start()
        t0 = time.time()
        if is_implicit:
            Ask.i("ir") << A.mul("ir",sk)
        else:
            Ask.i("ir") << A.i("irl")*sk.i("il")
        t1 = time.time()
        if ctf.comm().rank == 0 and status_prints == True:
            print('form Ask takes {}'.format(t1-t0))
        t_cg_bmvec.stop()

        Ask += regParam*sk

        rnorm = ctf.tensor(I)
        rnorm.i("i") << rk.i("ir") * rk.i("ir")
        
        skAsk = ctf.tensor(I)
        skAsk.i("i") << sk.i("ir") * Ask.i("ir")
        
        alpha = rnorm/(skAsk + 1.e-30)

        alphask = ctf.tensor((I,r))
        alphask.i("ir") << alpha.i("i") * sk.i("ir")
        xk1 = xk + alphask

        alphaask = ctf.tensor((I,r))
        alphaask.i("ir") << alpha.i("i") * Ask.i("ir")
        rk1 = rk - alphaask
        
        rk1norm = ctf.tensor(I)
        rk1norm.i("i") << rk1.i("ir") * rk1.i("ir")

        beta = rk1norm/(rnorm+ 1.e-30)

        betask = ctf.tensor((I,r))
        betask.i("ir") << beta.i("i") * sk.i("ir")
        sk1 = rk1 + betask
        rk = rk1
        xk = xk1
        sk = sk1
        if ctf.vecnorm(rk) < CG_thresh:
            break

    #print("explicit CG residual after",sk.shape[-1],"iterations is",ctf.vecnorm(rk))

    t_batch_cg.stop()
    return xk


def solve_SVD(A,b,factor,r,regParam):
    [U_,S_,VT_] = ctf.svd(A)
    S_ = 1/(S_+regParam*ctf.ones(S_.shape))
    factor.set_zero()
    factor.i("r") << VT_.i("kr")*S_.i("k")*U_.i("tk")*b.i("t")
    
    return factor   

def updateFactor(T,U,V,W,regParam,omega,I,J,K,r,num_blocks,string,use_implicit):

    t_RHS = ctf.timer("ALS_imp_cg_RHS")
    t_cg_TTTP = ctf.timer("ALS_imp_cg_TTTP")
    t_o_slice = ctf.timer("ALS_imp_omega_slice")
    t_form_EQs = ctf.timer("ALS_exp_form_EQs")
    t_form_RHS = ctf.timer("ALS_exp_form_RHS")
    if (string=="U"):

        block_size = int(I/num_blocks)
        for n in range(num_blocks):
            t_o_slice.start()
            nomega = omega[n*block_size : (n+1)*block_size,:,:]
            t_o_slice.stop()
            assert(nomega.sp == sparse_format)
            x0 = ctf.random.random((block_size,r))
            b = ctf.tensor((block_size,r))
            t_RHS.start()
            b.i("ir") << V.i("Jr")*W.i("Kr")*T[n*block_size : (n+1)*block_size,:,:].i("iJK")  # RHS; ATb
            t_RHS.stop()
        
            if use_implicit:
                Ax0 = ctf.tensor((block_size,r))
                t_cg_TTTP.start()
                Ax0.i("ir") << V.i("Jr")*W.i("Kr")*ctf.TTTP(nomega, [x0,V,W]).i("iJK")
                t_cg_TTTP.stop()
                Ax0 += regParam*x0 
                U[n*block_size : (n+1)*block_size,:] = CG(implicit_ATA(V,W,nomega,"U"),b,x0,r,regParam,block_size,True)
            else:
                A = ctf.tensor((block_size,r,r))
                t_form_EQs.start()
                A.i("iuv") << V.i("Ju")*W.i("Ku") * nomega.i("iJK")*V.i("Jv")*W.i("Kv")
                t_form_EQs.stop()
                U[n*block_size : (n+1)*block_size,:] = CG(A,b,x0,r,regParam,block_size)
         
        return U

    if (string=="V"):
        block_size = int(J/num_blocks)
        for n in range(num_blocks): 
            t_o_slice.start()
            nomega = omega[:,n*block_size : (n+1)*block_size,:]
            t_o_slice.stop()
            x0 = ctf.random.random((block_size,r))
            b = ctf.tensor((block_size,r))
            t_RHS.start()
            b.i("jr") << U.i("Ir")*W.i("Kr")*T[:,n*block_size : (n+1)*block_size,:].i("IjK")  # RHS; ATb
            t_RHS.stop()
            if use_implicit:
                Ax0 = ctf.tensor((block_size,r))
                t_cg_TTTP.start()
                Ax0.i("jr") << U.i("Ir")*W.i("Kr")*ctf.TTTP(nomega, [U,x0,W]).i("IjK")
                t_cg_TTTP.stop()
                Ax0 += regParam*x0
                V[n*block_size : (n+1)*block_size,:] = CG(implicit_ATA(U,W,nomega,"V"),b,x0,r,regParam,block_size,True)
            else:
                A = ctf.tensor((block_size,r,r))
                t_form_EQs.start()
                A.i("juv") << U.i("Iu")*W.i("Ku") * nomega.i("IjK") * U.i("Iv")*W.i("Kv") 
                t_form_EQs.stop()
                V[n*block_size : (n+1)*block_size,:] = CG(A,b,x0,r,regParam,block_size)

        return V  

    if (string=="W"):
        block_size = int(K/num_blocks)
        for n in range(num_blocks): 
            t_o_slice.start()
            nomega = omega[:,:,n*block_size : (n+1)*block_size]
            t_o_slice.stop()
            x0 = ctf.random.random((block_size,r))
            b = ctf.tensor((block_size,r))
            t_RHS.start()
            b.i("kr") << U.i("Ir")*V.i("Jr")* T[:,:,n*block_size : (n+1)*block_size].i("IJk")  # RHS; ATb
            t_RHS.stop()
            if use_implicit:
                Ax0 = ctf.tensor((block_size,r))
                t_cg_TTTP.start()
                Ax0.i("kr") << U.i("Ir")*V.i("Jr")*ctf.TTTP(nomega, [U,V,x0]).i("IJk")
                t_cg_TTTP.stop()
                Ax0 += regParam*x0 
                W[n*block_size : (n+1)*block_size,:] = CG(implicit_ATA(U,V,nomega,"W"),b,x0,r,regParam,block_size,True)
            else:
                A = ctf.tensor((block_size,r,r))
                t_form_EQs.start()
                A.i("kuv") << U.i("Iu")*V.i("Ju")*nomega.i("IJk")*U.i("Iv")*V.i("Jv")  # LHS; ATA using matrix-vector multiplication
                t_form_EQs.stop()
                W[n*block_size : (n+1)*block_size,:] = CG(A,b,x0,r,regParam,block_size)

        return W

def solve(A,b,factor,r,regParam):
    n = A.shape[0]
    L = ctf.cholesky(A+regParam*ctf.eye(n))
    factor = ctf.solve_tri(L,b.reshape((n,1)),True,True,False)
    factor = ctf.solve_tri(L,factor,True,True,True).reshape((n,))
    
    return factor   


def getALS_CG(T,U,V,W,regParam,omega,I,J,K,r,num_blocks,num_iter=100,err_thresh=.001,use_implicit=True):


    t_before_loop = time.time()

    if use_implicit == True:
        t_ALS_CG = ctf.timer_epoch("als_CG_implicit")
        if ctf.comm().rank() == 0:
            print("--------------------------------ALS with implicit CG------------------------")
    else:
        t_ALS_CG = ctf.timer_epoch("als_CG_explicit")
        if ctf.comm().rank() == 0:
            print("--------------------------------ALS with explicit CG------------------------")
    t_ALS_CG.begin()
 
    it = 0

    t_init_error_norm = ctf.timer("ALS_init_error_tensor_norm")
    t_init_error_norm.start()
    t0 = time.time()
    E = ctf.tensor((I,J,K),sp=sparse_format)
    #E.i("ijk") << T.i("ijk") - omega.i("ijk")*U.i("iu")*V.i("ju")*W.i("ku")
    E.i("ijk") << T.i("ijk") - ctf.TTTP(omega, [U,V,W]).i("ijk")
    assert(E.sp == sparse_format)
    t1 = time.time()
    curr_err_norm = ctf.vecnorm(E) + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W))*regParam
    t2= time.time()

    t_init_error_norm.stop()
    if ctf.comm().rank() == 0 and status_prints == True:
            print('ctf.TTTP() takes {}'.format(t1-t0))
            print('ctf.vecnorm {}'.format(t2-t1))
    
    ctf.random.seed(42)
    while True:

        t_upd_cg = ctf.timer("ALS_upd_cg")
        t_upd_cg.start()
        
        U = updateFactor(T,U,V,W,regParam,omega,I,J,K,r,num_blocks,"U",use_implicit)
        V = updateFactor(T,U,V,W,regParam,omega,I,J,K,r,num_blocks,"V",use_implicit) 
        W = updateFactor(T,U,V,W,regParam,omega,I,J,K,r,num_blocks,"W",use_implicit)
        
        E.set_zero()
        #E.i("ijk") << T.i("ijk") - omega.i("ijk")*U.i("iu")*V.i("ju")*W.i("ku")
        E.i("ijk") << T.i("ijk") - ctf.TTTP(omega, [U,V,W]).i("ijk")
        assert(E.sp == sparse_format)
        next_err_norm = ctf.vecnorm(E) + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W))*regParam

        t_upd_cg.stop()

        if ctf.comm().rank() == 0:
            print("Last residual:",curr_err_norm,"New residual",next_err_norm)
            it += 1
        
        if abs(curr_err_norm - next_err_norm) < err_thresh or it > num_iter:
            break

        curr_err_norm = next_err_norm

    t_ALS_CG.end()
    
    if glob_comm.rank() == 0:
        print('Time/Iteration: {}'.format((time.time() - t_before_loop)/1))
        print("Number of iterations: %d" % (it))
    



def getOmega_old(T,I,J,K):
    if (T.sp==False):
        omegactf = ((T > 0)*ctf.astensor(1.))
    else:
        omegactf = T / T
    
    return omegactf

def getOmega(T):
    [inds, data] = T.read_local_nnz()
    data[:] = 1.
    Omega = ctf.tensor(T.shape,sp=sparse_format)
    Omega.write(inds,data)
    return Omega


def main():

    I = 1000
    J = 1000
    K = 1000
    r = 2 
    sparsity = .000001
    regParam = .1
    num_blocks = 100
    num_iter = 20
    err_thresh = .001
    run_implicit = 1
    run_explicit = 1
    if len(sys.argv) >= 4:
        I = int(sys.argv[1])
        J = int(sys.argv[2])
        K = int(sys.argv[3])
    if len(sys.argv) >= 5:
        sparsity = np.float64(sys.argv[4])
    if len(sys.argv) >= 6:
        r = int(sys.argv[5])
    if len(sys.argv) >= 7:
        num_blocks = int(sys.argv[6])
    if len(sys.argv) >= 8:
        num_iter = int(sys.argv[7])
    if len(sys.argv) >= 9:
        err_thresh = np.float64(sys.argv[8])
    if len(sys.argv) >= 10:
        regParam= np.float64(sys.argv[9])
    if len(sys.argv) >= 11:
        run_implicit= int(sys.argv[10])
    if len(sys.argv) >= 12:
        run_explicit= int(sys.argv[11])

    if glob_comm.rank() == 0:
        print("I is",I,"J is",J,"K is",K,"sparisty is",sparsity,"r is",r,"num_blocks size is",num_blocks,"num_iter is",num_iter,"err_thresh is",err_thresh,"regParam is",regParam,"run_implicit",run_implicit,"run_explicit is",run_explicit)


    # 3rd-order tensor
    T_SVD = ctf.tensor((I,J,K),sp=sparse_format)
    T_SVD.fill_sp_random(0,1,sparsity)
    #T_SVD = function_tensor(I,J,K,sparsity)
    assert(T_SVD.sp == sparse_format)
 
    #omega = updateOmega(T_SVD,I,J,K)
    t0 = time.time()
    omega = getOmega(T_SVD)
    assert(omega.sp == sparse_format)
    if glob_comm.rank() == 0:
        print('getOmega takes {}'.format(time.time() - t0))
        
    ctf.random.seed(42)
    U_SVD = ctf.random.random((I,r))
    V_SVD = ctf.random.random((J,r))
    W_SVD = ctf.random.random((K,r))

    U_CG = ctf.copy(U_SVD)
    V_CG = ctf.copy(V_SVD)
    W_CG = ctf.copy(W_SVD)
    T_CG = ctf.copy(T_SVD)

    if run_implicit == True:
        getALS_CG(T_CG,U_CG,V_CG,W_CG,regParam,omega,I,J,K,r,num_blocks,num_iter,err_thresh,True) 
    if run_explicit == True:
        getALS_CG(T_SVD,U_SVD,V_SVD,W_SVD,regParam,omega,I,J,K,r,num_blocks,num_iter,err_thresh,False)



main()

