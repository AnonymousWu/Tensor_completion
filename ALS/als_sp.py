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

def CG(Ax0,b,x0,f1,f2,r,regParam,omega,I,string):

    t_cg_conv = ctf.timer("ALS_cg_conv")
    t_cg_conv.start()
    
    rk = b - Ax0
    sk = rk
    xk = x0
    for i in range(sk.shape[-1]): # how many iterations?
        Ask = ctf.tensor((I,r))
        t_cg_TTTP = ctf.timer("ALS_cg_TTTP")
        t_cg_TTTP.start()
        if (string=="U"):
            t0 = time.time()
            #Ask.i("ir") << f1.i("Jr")*f2.i("Kr")*omega.i("iJK")*f1.i("JR")*f2.i("KR")*sk.i("iR")
            Ask.i("ir") << f1.i("Jr")*f2.i("Kr")*ctf.TTTP(omega,[sk,f1,f2]).i("iJK")
            t1 = time.time()
            if ctf.comm().rank == 0 and status_prints == True:
                print('form Ask takes {}'.format(t1-t0))
        if (string=="V"):
            #Ask.i("jr") << f1.i("Ir")*f2.i("Kr")*omega.i("IjK")*f1.i("IR")*f2.i("KR")*sk.i("jR")
            Ask.i("jr") << f1.i("Ir")*f2.i("Kr")*ctf.TTTP(omega,[f1,sk,f2]).i("IjK")
        if (string=="W"):
            #Ask.i("kr") << f1.i("Ir")*f2.i("Jr")*omega.i("IJk")*f1.i("IR")*f2.i("JR")*sk.i("kR")
            Ask.i("kr") << f1.i("Ir")*f2.i("Jr")*ctf.TTTP(omega,[f1,f2,sk]).i("IJk")
        t_cg_TTTP.stop()

        Ask += regParam*sk

        rnorm = ctf.tensor(I)
        rnorm.i("i") << rk.i("ir") * rk.i("ir")
        #print("rnorm",rnorm.to_nparray())

        #for i in range(I):
        #    if rnorm[i] < 1.e-16:

        #    break
        
        skAsk = ctf.tensor(I)
        skAsk.i("i") << sk.i("ir") * Ask.i("ir")
        
        #if (rnorm[i] < 1.e-30):
        #    continue
        alpha = rnorm/(skAsk + 1.e-30)

        alphask = ctf.tensor((I,r))
        alphask.i("ir") << alpha.i("i") * sk.i("ir")
        xk1 = xk + alphask

        alphaask = ctf.tensor((I,r))
        alphaask.i("ir") << alpha.i("i") * Ask.i("ir")
        rk1 = rk - alphaask
        
        rk1norm = ctf.tensor(I)
        rk1norm.i("i") << rk1.i("ir") * rk1.i("ir")

        #if (rk1norm[i] < 1.e-30):
        #    continue
        beta = rk1norm/(rnorm+ 1.e-30)

        betask = ctf.tensor((I,r))
        betask.i("ir") << beta.i("i") * sk.i("ir")
        sk1 = rk1 + betask
        rk = rk1
        xk = xk1
        sk = sk1
        if ctf.vecnorm(rk) < CG_thresh:
            break

    #print("CG residual after",sk.shape[-1],"iterations is",ctf.vecnorm(rk))

    t_cg_conv.stop()
    return xk


def updateFactor_CG(T,U,V,W,regParam,omega,I,J,K,r,block,string):

    t_RHS = ctf.timer("ALS_cg_RHS")
    t_cg_TTTP = ctf.timer("ALS_cg_TTTP")
    t_o_slice = ctf.timer("ALS_omega_slice")
    if (string=="U"):

        size = int(I/block)
        for n in range(block):

            t0 = time.time()
            t_o_slice.start()
            nomega = omega[n*size : (n+1)*size,:,:]
            t_o_slice.stop()
            assert(nomega.sp == sparse_format)
            t1 = time.time()
        
            x0 = ctf.random.random((size,r))
            Ax0 = ctf.tensor((size,r))
            t_cg_TTTP.start()
            #Ax0.i("ir") << V.i("Jr")*W.i("Kr")*nomega.i("iJK")*V.i("JR")*W.i("KR")*x0.i("iR")  # LHS; ATA using matrix-vector multiplication
            Ax0.i("ir") << V.i("Jr")*W.i("Kr")*ctf.TTTP(nomega, [x0,V,W]).i("iJK")
            t_cg_TTTP.stop()
            Ax0 += regParam*x0 
            t2 = time.time()

            b = ctf.tensor((size,r))
            t_RHS.start()
            b.i("ir") << V.i("Jr")*W.i("Kr")*T[n*size : (n+1)*size,:,:].i("iJK")  # RHS; ATb
            t_RHS.stop()
            t3 = time.time()

            if ctf.comm().rank() == 0 and status_prints == True:
                print('slicing Omega takes {}'.format(t1 - t0))
                print('form Ax0 takes {}'.format(t2 - t1))
                print('form b takes {}'.format(t3 - t2))
        
            #U[n*size : (n+1)*size,:].set_zero()
            U[n*size : (n+1)*size,:] = CG(Ax0,b,x0,V,W,r,regParam,nomega,size,"U")
     
        return U

    if (string=="V"):
        size = int(J/block)
        for n in range(block): 
            t_o_slice.start()
            nomega = omega[:,n*size : (n+1)*size,:]
            t_o_slice.stop()
            x0 = ctf.random.random((size,r))
            Ax0 = ctf.tensor((size,r))
            t_cg_TTTP.start()
            #Ax0.i("jr") << U.i("Ir")*W.i("Kr")*nomega.i("IjK")*U.i("IR")*W.i("KR")*x0.i("jR")  # LHS; ATA using matrix-vector multiplication
            Ax0.i("jr") << U.i("Ir")*W.i("Kr")*ctf.TTTP(nomega, [U,x0,W]).i("IjK")
            t_cg_TTTP.stop()
            Ax0 += regParam*x0 
            b = ctf.tensor((size,r))
            t_RHS.start()
            b.i("jr") << U.i("Ir")*W.i("Kr")*T[:,n*size : (n+1)*size,:].i("IjK")  # RHS; ATb
            t_RHS.stop()
            #V[n*size : (n+1)*size,:].set_zero()
            V[n*size : (n+1)*size,:] = CG(Ax0,b,x0,U,W,r,regParam,nomega,size,"V")

    
        return V  

    if (string=="W"):
        size = int(K/block)
        for n in range(block): 
            t_o_slice.start()
            nomega = omega[:,:,n*size : (n+1)*size]
            t_o_slice.stop()
            x0 = ctf.random.random((size,r))
            Ax0 = ctf.tensor((size,r))
            t_cg_TTTP.start()
            #Ax0.i("kr") << U.i("Ir")*V.i("Jr")*nomega.i("IJk")*U.i("IR")*V.i("JR")*x0.i("kR")  # LHS; ATA using matrix-vector multiplication
            Ax0.i("kr") << U.i("Ir")*V.i("Jr")*ctf.TTTP(nomega, [U,V,x0]).i("IJk")
            t_cg_TTTP.stop()
            Ax0 += regParam*x0 
            b = ctf.tensor((size,r))
            t_RHS.start()
            b.i("kr") << U.i("Ir")*V.i("Jr")* T[:,:,n*size : (n+1)*size].i("IJk")  # RHS; ATb
            t_RHS.stop()
            W[n*size : (n+1)*size,:] = CG(Ax0,b,x0,U,V,r,regParam,nomega,size,"W")

        return W


def Kressner_SVD(A,b,factor,r,regParam):
    [U_,S_,VT_] = ctf.svd(A)
    S_ = 1/(S_+regParam*ctf.ones(S_.shape))
    factor.set_zero()
    factor.i("r") << VT_.i("kr")*S_.i("k")*U_.i("tk")*b.i("t")
    
    return factor   

def Kressner(A,b,factor,r,regParam):
    n = A.shape[0]
    L = ctf.cholesky(A+regParam*ctf.eye(n))
    factor = ctf.solve_tri(L,b.reshape((n,1)),True,True,False)
    factor = ctf.solve_tri(L,factor,True,True,True).reshape((n,))
    
    #[U_,S_,VT_] = ctf.svd(A)
    #S_ = 1/(S_+regParam*ctf.ones(S_.shape))
    #factor.set_zero()
    #factor.i("r") << VT_.i("kr")*S_.i("k")*U_.i("tk")*b.i("t")
    
    return factor   


def updateFactor_Kressner(T,U,V,W,regParam,omega,I,J,K,r,string):

    t = time.time()

    t_form_EQs = ctf.timer("ALS_form_EQs")
    t_form_RHS = ctf.timer("ALS_form_RHS")

    if (string=="U"):
        for i in range(I):
            A = ctf.tensor((r,r))
            t_form_EQs.start()
            A.i("uv") << V.i("Ju")*W.i("Ku") * omega[i,:,:].i("JK")*V.i("Jv")*W.i("Kv")
            t_form_EQs.stop()

            b = ctf.tensor(r)
            sliced_T = T[i,:,:]
            t_form_RHS.start()
            b.i("r") << V.i("Jr")*W.i("Kr") * sliced_T.i("JK")  # RHS; ATb
            t_form_RHS.stop()
            U[i,:]= Kressner(A,b,U[i,:],r,regParam) 

        return U

    if (string=="V"):
        for j in range(J):
            A = ctf.tensor((r,r))
            t_form_EQs.start()
            A.i("uv") << U.i("Iu")*W.i("Ku") * omega[:,j,:].i("IK") * U.i("Iv")*W.i("Kv") 
            t_form_EQs.stop()
            b = ctf.tensor(r)
            t_form_RHS.start()
            b.i("r") <<  U.i("Ir")*W.i("Kr") * T[:,j,:].i("IK")  # RHS; ATb
            t_form_EQs.stop()
            V[j,:] = Kressner(A,b,V[j,:],r,regParam)

        return V  

    if (string=="W"):
        for k in range(K):
            A= ctf.tensor((r,r))
            t_form_EQs.start()
            A.i("uv") << U.i("Iu")*V.i("Ju")*omega[:,:,k].i("IJ")*U.i("Iv")*V.i("Jv")  # LHS; ATA using matrix-vector multiplication
            t_form_EQs.stop()
            b = ctf.tensor(r)
            t_form_RHS.start()
            b.i("r") << U.i("Ir")*V.i("Jr") * T[:,:,k].i("IJ")  # RHS; ATb
            t_form_RHS.stop()
            W[k,:] = Kressner(A,b,W[k,:],r,regParam) 

        return W


def getALS_CG(T,U,V,W,regParam,omega,I,J,K,r,block,num_iter=100,err_thresh=.001):

    if ctf.comm().rank() == 0:
        print("--------------------------------ALS iterative CG------------------------")

    t_before_loop = time.time()

    t_ALS_CG = ctf.timer_epoch("als_CG")
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
    
    while True:

        t_upd_cg = ctf.timer("ALS_upd_cg")
        t_upd_cg.start()
        
        U = updateFactor_CG(T,U,V,W,regParam,omega,I,J,K,r,block,"U")
        V = updateFactor_CG(T,U,V,W,regParam,omega,I,J,K,r,block,"V") 
        W = updateFactor_CG(T,U,V,W,regParam,omega,I,J,K,r,block,"W")
        
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
    


def getALS_Kressner(T,U,V,W,regParam,omega,I,J,K,r,num_iter=100,err_thresh=.001):

    if ctf.comm().rank() == 0:
        print("--------------------------------ALS direct SVD------------------------")
    t_ALS_Kressner = ctf.timer_epoch("als_Kressner")
    t_ALS_Kressner.begin()

    t_before_loop = time.time()
    it = 0
    E = ctf.tensor((I,J,K),sp=sparse_format)
    #E.i("ijk") << T.i("ijk") - omega.i("ijk")*U.i("iu")*V.i("ju")*W.i("ku")
    E.i("ijk") << T.i("ijk") - ctf.TTTP(omega, [U,V,W]).i("ijk")
    assert(E.sp == sparse_format)
    curr_err_norm = ctf.vecnorm(E) + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W))*regParam
    
    while True:

        t = time.time()

        U = updateFactor_Kressner(T,U,V,W,regParam,omega,I,J,K,r,"U")
        V = updateFactor_Kressner(T,U,V,W,regParam,omega,I,J,K,r,"V") 
        W = updateFactor_Kressner(T,U,V,W,regParam,omega,I,J,K,r,"W")
        
        E.set_zero()
        #E.i("ijk") << T.i("ijk") - omega.i("ijk")*U.i("iu")*V.i("ju")*W.i("ku")
        E.i("ijk") << T.i("ijk") - ctf.TTTP(omega, [U,V,W]).i("ijk")
        assert(E.sp == sparse_format)
        next_err_norm = ctf.vecnorm(E) + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W))*regParam
            
        if ctf.comm().rank() == 0:
            print("Last residual:",curr_err_norm,"New residual",next_err_norm)
            it +=1
       
        if abs(curr_err_norm - next_err_norm) < err_thresh or it > num_iter:
            break
        curr_err_norm = next_err_norm
       
    t_ALS_Kressner.end()
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
    block = 100
    num_iter = 20
    err_thresh = .001
    if len(sys.argv) >= 4:
        I = int(sys.argv[1])
        J = int(sys.argv[2])
        K = int(sys.argv[3])
    if len(sys.argv) >= 5:
        sparsity = np.float64(sys.argv[4])
    if len(sys.argv) >= 6:
        r = int(sys.argv[5])
    if len(sys.argv) >= 7:
        block = int(sys.argv[6])
    if len(sys.argv) >= 8:
        num_iter = int(sys.argv[7])
    if len(sys.argv) >= 9:
        err_thresh = np.float64(sys.argv[8])
    if len(sys.argv) >= 10:
        regParam= np.float64(sys.argv[9])

    if glob_comm.rank() == 0:
        print("I is",I,"J is",J,"K is",K,"sparisty is",sparsity,"r is",r,"block size is",block,"num_iter is",num_iter,"err_thresh is",err_thresh,"regParam is",regParam)


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


    getALS_CG(T_CG,U_CG,V_CG,W_CG,regParam,omega,I,J,K,r,block,num_iter,err_thresh) 
    getALS_Kressner(T_SVD,U_SVD,V_SVD,W_SVD,regParam,omega,I,J,K,r,num_iter,err_thresh)



main()

