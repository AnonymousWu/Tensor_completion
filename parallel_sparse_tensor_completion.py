import ctf,time,random
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from ctf import random as crandom
glob_comm = ctf.comm()
#from scipy.sparse.linalg import lsqr as lsqr
from function_tensor import *
from sklearn.metrics import mean_squared_error
from math import sqrt

class UnitTests:
        
    def test_3d_purturb1(self):
        
        I = random.randint(3,5)
        J = random.randint(3,5)
        K = random.randint(3,5)
        r = 2 
        sparsity = .2
        regParam = 10
        
        ctf.random.seed(42)
        U = ctf.random.random((I,r))
        V= ctf.random.random((J,r))
        W= ctf.random.random((K,r))
    
    
        # 3rd-order tensor
        T = ctf.tensor((I,J,K))
        T.fill_random(0,1)
        omega = createOmega(I,J,K,sparsity)
        T.i("ijk") << omega.i("ijk")*U.i("iu")*V.i("ju")*W.i("ku")
        omega = updateOmega(T,I,J,K)
        
        print("U: ",U)
                
        # purturb the first factor matrix
        U += crandom.random((I,r))*.01
        # call updateU function
        nU = updateU(T,U,V,W,regParam,omega,I,J,K,r)
        
        print("nU: ",nU)
        
        nT = ctf.tensor((I,J,K))
        nT.i("ijk") << omega.i("ijk")*nU.i("iu")*V.i("ju")*W.i("ku")
        
        print("nT: ",nT)
        print("T: ",T)
    
        assert(ctf.all(ctf.abs(nT - T < 1e-10)))
        print("passed test: test_3d_purturb1")

        
    def runAllTests(self):
        self.test_3d_purturb1()


def normalize(Z,r):
    norms = ctf.tensor(r)
    norms.i("u") << Z.i("pu")*Z.i("pu")
    norms = 1./norms**.5
    X = ctf.tensor(copy=Z)
    Z.set_zero()
    Z.i("pu") << X.i("pu")*norms.i("u")
    return 1./norms



def createOmega(I,J,K,sparsity):
    print(I,J,K)
    #Actf = ctf.tensor((I,J,K),sp=True)
    #Actf.fill_sp_random(0,1,sparsity)
    Actf = ctf.tensor((I,J,K))
    Actf.fill_random(0,1)
    omegactf = ((Actf > 0)*ctf.astensor(1.))
    return omegactf


def updateOmega(T,I,J,K):
    '''
    Gets a random subset of rows for each U,V,W iteration
    '''
    if (T.sp==False):
        omegactf = ((T > 0)*ctf.astensor(1.))
    else:
        omegactf = T / T
    
    return omegactf

def getDenseOmega(T,U,V,W,regParam,omega,I,J,K,r,idx,string):
    if (string =="i"):
        omega_curr = ctf.to_nparray(omega[idx,:,:].reshape(J*K))
        omega_sum = np.cumsum(omega_curr).tolist()
        omega_sum.insert(0,0)
        del omega_sum[-1]
        #print("omega prefix sum: ", omega_sum)
        l = []
        for x,y in enumerate(omega_sum):
            if omega_curr[x] != 0:
                l.append((x,int(y)))
        #print(l)
        num_nonzero = len(l)
        
        # form dense omega matrix
        temp = np.zeros((J*K,len(l)))
        for x,y in l:
            temp[x][y] = 1
        #print("omega_dense: ", omega_dense)
       
        omega_dense = ctf.astensor(temp)
        #print("before", (omega_dense, omega_dense.shape))
        omega_dense = omega_dense.reshape(J,K,num_nonzero)
        #print("after", (omega_dense, omega_dense.shape))       
    
    if (string =="j"):
        omega_curr = ctf.to_nparray(omega[:,idx,:].reshape(I*K))
        omega_sum = np.cumsum(omega_curr).tolist()
        omega_sum.insert(0,0)
        del omega_sum[-1]
        l = []
        for x,y in enumerate(omega_sum):
            if omega_curr[x] != 0:
                l.append((x,int(y)))
        num_nonzero = len(l)
        temp = np.zeros((I*K,len(l)))
        for x,y in l:
            temp[x,y] = 1
        omega_dense = ctf.astensor(temp)
        omega_dense = omega_dense.reshape(I,K,num_nonzero)        
    
    if (string =="k"):
        omega_curr = ctf.to_nparray(omega[:,:,idx].reshape(I*J))
        omega_sum = np.cumsum(omega_curr).tolist()
        omega_sum.insert(0,0)
        del omega_sum[-1]
        l = []
        for x,y in enumerate(omega_sum):
            if omega_curr[x] != 0:
                l.append((x,int(y)))
        num_nonzero = len(l)  
        temp = np.zeros((I*J,len(l)))
        for x,y in l:
            temp[x][y] = 1
        omega_dense = ctf.astensor(temp)
        omega_dense = omega_dense.reshape(I,J,num_nonzero)
        
    return num_nonzero,omega_dense



def getDenseOmega_all(T,U,V,W,regParam,omega,I,J,K,r,string):
    if (string =="i"):
        omega_dense = []
        for idx in range(I):
            omega_curr = ctf.to_nparray(omega[idx,:,:].reshape(J*K))
            omega_sum = np.cumsum(omega_curr).tolist()
            omega_sum.insert(0,0)
            del omega_sum[-1]
            #print("omega prefix sum: ", omega_sum)
            l = []
            for x,y in enumerate(omega_sum):
                if omega_curr[x] != 0:
                    l.append((x,int(y)))
            #print(l)
            num_nonzero = len(l)
        
            # form dense omega matrix
            #temp = np.zeros((J*K,len(l)))
            temp = np.zeros((J*K,J*K))
            for x,y in l:
                temp[x][y] = 1
            #print("omega_dense: ", omega_dense)
            temp = np.reshape(temp,(J,K,J*K))
            omega_dense.append(temp)
        omega_dense = ctf.astensor(np.stack(omega_dense,axis=3))
        #print("omega_dense shape: ", omega_dense.shape)

        return num_nonzero,omega_dense   


def LS_SVD(Z,factor,r,Tbar,regParam):
    [U_,S_,VT_] = ctf.svd(Z)
    S_ = S_/(S_*S_ + regParam*ctf.ones(S_.shape))
    factor.set_zero()
    factor.i("r") << VT_.i("kr")*S_.i("k")*U_.i("tk")*Tbar.i("t")
    
    return factor    

def updateFactor_SVD(T,U,V,W,regParam,omega,I,J,K,r,string):

    if (string=="U"):

        M1 = ctf.tensor((J,K,r))
        M1.i("jku") << V.i("ju")*W.i("ku")
    
        for i in range(I):
            num_nonzero, dense_omega = getDenseOmega(T,U,V,W,regParam,omega,I,J,K,r,i,"i")

            Z = ctf.tensor((num_nonzero,r))
            Z.i("tr") << dense_omega.i("jkt")*M1.i("jkr")
        
            Tbar = ctf.tensor((num_nonzero))
            Tbar.i("t") << dense_omega.i("jkt") *T[i,:,:].i("jk")
        
            U[i,:].set_zero()
            U[i,:] = LS_SVD(Z,U[i,:],r,Tbar,regParam)

        return U

    if (string == 'V'):

        M2 = ctf.tensor((I,K,r))
        M2.i("iku") << U.i("iu")*W.i("ku")
    
        for j in range(J):
            num_nonzero, dense_omega = getDenseOmega(T,U,V,W,regParam,omega,I,J,K,r,j,"j")
            Z = ctf.tensor((num_nonzero,r))
            Z.i("tr") << dense_omega.i("ikt")*M2.i("ikr")
        
            Tbar = ctf.tensor((num_nonzero))
            Tbar.i("t") << dense_omega.i("ikt") *T[:,j,:].i("ik")
        
            V[j,:].set_zero()
            V[j,:] = LS_SVD(Z,V[j,:],r,Tbar,regParam)

        return V

    if (string == 'W'):

        M3 = ctf.tensor((I,J,r))
        M3.i("iju") << U.i("iu")*V.i("ju")
    
        for k in range(K):
            num_nonzero, dense_omega = getDenseOmega(T,U,V,W,regParam,omega,I,J,K,r,k,"k")
            Z = ctf.tensor((num_nonzero,r))
            Z.i("tr") << dense_omega.i("ijt")*M3.i("ijr")
        
            Tbar = ctf.tensor((num_nonzero))
            Tbar.i("t") << dense_omega.i("ijt") *T[:,:,k].i("ij")
        
            W[k,:].set_zero()
            W[k,:] = LS_SVD(Z,W[k,:],r,Tbar,regParam)
       
        return W


def CG(Ax0,b,x0,f1,f2,r,regParam,omega,I,string):
    rk = b - Ax0
    sk = rk
    xk = x0
    for i in range(sk.shape[-1]): # how many iterations?
        Ask = ctf.tensor((I,r))
        if (string=="U"):
            Ask.i("ir") << f1.i("Jr")*f2.i("Kr")*omega.i("iJK")*f1.i("JR")*f2.i("KR")*sk.i("iR")
        if (string=="V"):
            Ask.i("jr") << f1.i("Ir")*f2.i("Kr")*omega.i("IjK")*f1.i("IR")*f2.i("KR")*sk.i("jR")
        if (string=="W"):
            Ask.i("kr") << f1.i("Ir")*f2.i("Jr")*omega.i("IJk")*f1.i("IR")*f2.i("JR")*sk.i("kR")

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
        #print("rk",ctf.vecnorm(rk))
    return xk


def updateFactor_CG(T,U,V,W,regParam,omega,I,J,K,r,block,string):

    if (string=="U"):
        #M1 = ctf.tensor((J,K,r))
        #M1.i("jku") << V.i("ju")*W.i("ku")
    
        #num_nonzero, dense_omega = getDenseOmega(T,U,V,W,regParam,omega,I,J,K,r,"i")
        #Z = ctf.tensor((I,J*K,r))
        #Z.i("itr") << dense_omega.i("jkti")*M1.i("jkr")
        #Tbar = ctf.tensor((I,num_nonzero))
        #Tbar.i("it") << dense_omega.i("ijkt") *T.i("ijk")

        size = int(I/block)
        for n in range(block): 
            nomega = omega[n*size : (n+1)*size,:,:]
            x0 = ctf.random.random((size,r))
            Ax0 = ctf.tensor((size,r))
            #Ax0.i("ir") << M.i("jkr")*dense_omega.i("jkti")*dense_omega.i("jktI")*M.i("jkR")*x0.i("IR")
            Ax0.i("ir") << V.i("Jr")*W.i("Kr")*nomega.i("iJK")*V.i("JR")*W.i("KR")*x0.i("iR")  # LHS; ATA using matrix-vector multiplication
            Ax0 += regParam*x0 
            #print("Ax0: ",Ax0)

            b = ctf.tensor((size,r))
            #b.i("ir") << M.i("JKr") * dense_omega.i("JKti") * dense_omega.i("JKtI") * T.i("IJK")
            b.i("ir") << V.i("Jr")*W.i("Kr")*T[n*size : (n+1)*size,:,:].i("iJK")  # RHS; ATb
            #print("b: ",b)
        
            U[n*size : (n+1)*size,:].set_zero()
            U[n*size : (n+1)*size,:] = CG(Ax0,b,x0,V,W,r,regParam,nomega,size,"U")
     
        return U

    if (string=="V"):
        #M2 = ctf.tensor((I,K,r))
        #M2.i("iku") << U.i("iu")*W.i("ku")
    
        #num_nonzero, dense_omega = getDenseOmega(T,U,V,W,regParam,omega,I,J,K,r)
        #Z = ctf.tensor((J,num_nonzero,r))
        #Z.i("jtr") << dense_omega.i("ijkt")*M2.i("ikr")     
        #Tbar = ctf.tensor((J,num_nonzero))
        #Tbar.i("jt") << dense_omega.i("ijkt") *T.i("ijk")   

        size = int(J/block)
        for n in range(block): 
            nomega = omega[:,n*size : (n+1)*size,:]
            x0 = ctf.random.random((size,r))
            Ax0 = ctf.tensor((size,r))
            Ax0.i("jr") << U.i("Ir")*W.i("Kr")*nomega.i("IjK")*U.i("IR")*W.i("KR")*x0.i("jR")  # LHS; ATA using matrix-vector multiplication
            Ax0 += regParam*x0 
            b = ctf.tensor((size,r))
            b.i("jr") << U.i("Ir")*W.i("Kr")*T[:,n*size : (n+1)*size,:].i("IjK")  # RHS; ATb
            V[n*size : (n+1)*size,:].set_zero()
            V[n*size : (n+1)*size,:] = CG(Ax0,b,x0,U,W,r,regParam,nomega,size,"V")
    
        return V  

    if (string=="W"):
        #M3 = ctf.tensor((I,J,r))
        #M3.i("iju") << U.i("iu")*V.i("ju")

        #num_nonzero, dense_omega = getDenseOmega(T,U,V,W,regParam,omega,I,J,K,r)
        #Z = ctf.tensor((K,num_nonzero,r))
        #Z.i("ktr") << dense_omega.i("ijkt")*M3.i("ijr")
        
        #Tbar = ctf.tensor((K,num_nonzero))
        #Tbar.i("kt") << dense_omega.i("ijkt") *T.i("ijk")

        size = int(K/block)
        for n in range(block): 
            nomega = omega[:,:,n*size : (n+1)*size]
            x0 = ctf.random.random((size,r))
            Ax0 = ctf.tensor((size,r))
            Ax0.i("kr") << U.i("Ir")*V.i("Jr")*nomega.i("IJk")*U.i("IR")*V.i("JR")*x0.i("kR")  # LHS; ATA using matrix-vector multiplication
            Ax0 += regParam*x0 
            b = ctf.tensor((size,r))
            b.i("kr") << U.i("Ir")*V.i("Jr")* T[:,:,n*size : (n+1)*size].i("IJk")  # RHS; ATb
            W[n*size : (n+1)*size,:].set_zero()
            W[n*size : (n+1)*size,:] = CG(Ax0,b,x0,U,V,r,regParam,nomega,size,"W")
    
        return W


def Kressner(A,b,factor,r,regParam):
    [U_,S_,VT_] = ctf.svd(A)
    S_ = 1/(S_+regParam*ctf.ones(S_.shape))
    factor.set_zero()
    factor.i("r") << VT_.i("kr")*S_.i("k")*U_.i("tk")*b.i("t")
    
    return factor   


def updateFactor_Kressner(T,U,V,W,regParam,omega,I,J,K,r,string):

    if (string=="U"):
        #M1 = ctf.tensor((J,K,r))
        #M1.i("jku") << V.i("ju")*W.i("ku")
        for i in range(I):
            A = ctf.tensor((r,r))
            A.i("uv") << V.i("Ju")*W.i("Ku") * omega[i,:,:].i("JK")*V.i("Jv")*W.i("Kv")
            #print("A: ",A)
            b = ctf.tensor(r)
            b.i("r") << V.i("Jr")*W.i("Kr") * T[i,:,:].i("JK")  # RHS; ATb
            #print("b: ",b)    
            U[i,:].set_zero()
            U[i,:]= Kressner(A,b,U[i,:],r,regParam)     
        return U

    if (string=="V"):
        #M2 = ctf.tensor((I,K,r))
        #M2.i("iku") << U.i("iu")*W.i("ku")
        for j in range(J):
            A = ctf.tensor((r,r))
            A.i("uv") << U.i("Iu")*W.i("Ku") * omega[:,j,:].i("IK") * U.i("Iv")*W.i("Kv")      
            b = ctf.tensor(r)
            b.i("r") <<  U.i("Ir")*W.i("Kr") * T[:,j,:].i("IK")  # RHS; ATb
            V[j,:].set_zero()
            V[j,:] = Kressner(A,b,V[j,:],r,regParam)    
        return V  

    if (string=="W"):
        #M3 = ctf.tensor((I,J,r))
        #M3.i("iju") << U.i("iu")*V.i("ju")
        for k in range(K):
            A= ctf.tensor((r,r))
            A.i("uv") << U.i("Iu")*V.i("Ju")*omega[:,:,k].i("IJ")*U.i("Iv")*V.i("Jv")  # LHS; ATA using matrix-vector multiplication
            b = ctf.tensor(r)
            b.i("r") << U.i("Ir")*V.i("Jr") * T[:,:,k].i("IJ")  # RHS; ATb
            W[k,:].set_zero()
            W[k,:] = Kressner(A,b,W[k,:],r,regParam) 
        return W



def getALS_SVD(T,U,V,W,regParam,omega,I,J,K,r):
   
    it = 0
    E = ctf.tensor((I,J,K))
    E.i("ijk") << T.i("ijk") - omega.i("ijk")*U.i("iu")*V.i("ju")*W.i("ku")
    curr_err_norm = ctf.vecnorm(E) + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W))*regParam
    
    while True:
        
        U = updateFactor_SVD(T,U,V,W,regParam,omega,I,J,K,r,"U")
        V = updateFactor_SVD(T,U,V,W,regParam,omega,I,J,K,r,"V") 
        W = updateFactor_SVD(T,U,V,W,regParam,omega,I,J,K,r,"W")
        
        E.set_zero()
        E.i("ijk") << T.i("ijk") - omega.i("ijk")*U.i("iu")*V.i("ju")*W.i("ku")
        next_err_norm = ctf.vecnorm(E) + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W))*regParam
            
        print(curr_err_norm, next_err_norm)
        
        if abs(curr_err_norm - next_err_norm) < .001 or it > 20:
            break
        curr_err_norm = next_err_norm
        it += 1
    
    print("Number of iterations: ", it)
    return U,V,W


def getALS_CG(T,U,V,W,regParam,omega,I,J,K,r,block):
 
    it = 0
    E = ctf.tensor((I,J,K))
    E.i("ijk") << T.i("ijk") - omega.i("ijk")*U.i("iu")*V.i("ju")*W.i("ku")
    curr_err_norm = ctf.vecnorm(E) + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W))*regParam
    norm = [curr_err_norm]
    timeList = [0]
    t= time.time()
    
    while True:

        U = updateFactor_CG(T,U,V,W,regParam,omega,I,J,K,r,block,"U")
        V = updateFactor_CG(T,U,V,W,regParam,omega,I,J,K,r,block,"V") 
        W = updateFactor_CG(T,U,V,W,regParam,omega,I,J,K,r,block,"W")
        
        E.set_zero()
        E.i("ijk") << T.i("ijk") - omega.i("ijk")*U.i("iu")*V.i("ju")*W.i("ku")
        next_err_norm = ctf.vecnorm(E) + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W))*regParam
            
        if ctf.comm().rank() == 0:
            print(curr_err_norm, next_err_norm)
        
        if abs(curr_err_norm - next_err_norm) < .001 or it > 100:
            break

        curr_err_norm = next_err_norm
        norm.append(curr_err_norm)
        timeList.append(np.round_(time.time()- t,4))
        it += 1
    
    return norm, it, timeList


def getALS_Kressner(T,U,V,W,regParam,omega,I,J,K,r):

    it = 0
    E = ctf.tensor((I,J,K))
    E.i("ijk") << T.i("ijk") - omega.i("ijk")*U.i("iu")*V.i("ju")*W.i("ku")
    curr_err_norm = ctf.vecnorm(E) + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W))*regParam
    norm = [curr_err_norm]
    timeList = [0]
    t= time.time()
    
    while True:

        U = updateFactor_Kressner(T,U,V,W,regParam,omega,I,J,K,r,"U")
        V = updateFactor_Kressner(T,U,V,W,regParam,omega,I,J,K,r,"V") 
        W = updateFactor_Kressner(T,U,V,W,regParam,omega,I,J,K,r,"W")
        
        E.set_zero()
        E.i("ijk") << T.i("ijk") - omega.i("ijk")*U.i("iu")*V.i("ju")*W.i("ku")
        next_err_norm = ctf.vecnorm(E) + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W))*regParam
            
        if ctf.comm().rank() == 0:
            print(curr_err_norm, next_err_norm)
        
        if abs(curr_err_norm - next_err_norm) < .001 or it > 100:
            break
        curr_err_norm = next_err_norm
        norm.append(curr_err_norm)
        timeList.append(np.round_(time.time()- t,4))
        it += 1
    
    return norm, it, timeList


def main():
    
    #ut = UnitTests()
    #ut.runAllTests()

    #I = random.randint(6,6)
    #J = random.randint(6,6)
    #K = random.randint(6,6)
    I = 4
    J = 4
    K = 4
    r = 2 
    sparsity = .1
    regParam = .1
    block = 4
    ntrails = 5

    # 3rd-order tensor
    #T_SVD = ctf.tensor((I,J,K),sp=True)
    #T_SVD.fill_sp_random(0,1,sparsity)
    T_SVD = function_tensor(I,J,K,sparsity)
 
    omega = updateOmega(T_SVD,I,J,K)


    blockCGerrList = []
    blockCGtimeList = []
    KressnererrList = []
    KressnertimeList = []
    for i in range(ntrails):   
        
        ctf.random.seed(42+i)
        U_SVD = ctf.random.random((I,r))
        V_SVD = ctf.random.random((J,r))
        W_SVD = ctf.random.random((K,r))

        U_CG = ctf.copy(U_SVD)
        V_CG = ctf.copy(V_SVD)
        W_CG = ctf.copy(W_SVD)
        T_CG = ctf.copy(T_SVD)

        U_CG2 = ctf.copy(U_SVD)
        V_CG2 = ctf.copy(V_SVD)
        W_CG2 = ctf.copy(W_SVD)
        T_CG2 = ctf.copy(T_SVD)
        
    #t = time.time()  
    #getALS_SVD(T_SVD,U_SVD,V_SVD,W_SVD,regParam,omega,I,J,K,r)   
    #print("ALS SVD costs time = ",np.round_(time.time()- t,4))    

        t = time.time()
        blockCGnorm,blockCGit,blockCGtime = getALS_CG(T_CG,U_CG,V_CG,W_CG,regParam,omega,I,J,K,r,block)
        blockCGerrList.append(blockCGnorm)
        blockCGtimeList.append(blockCGtime)
        if ctf.comm().rank() == 0:
            print("Number of iterations: ", blockCGit)
            print("CG block size = ",block)   
            print("ALS iterative CG costs time = ",np.round_(time.time()- t,4))  

        t = time.time()
        kressnernorm,kressnerit,kressnertime = getALS_Kressner(T_CG2,U_CG2,V_CG2,W_CG2,regParam,omega,I,J,K,r)
        KressnererrList.append(kressnernorm)
        KressnertimeList.append(kressnertime)
        if ctf.comm().rank() == 0:
            print("Number of iterations: ", kressnerit)
            print("ALS direct CG costs time = ",np.round_(time.time()- t,4))     


#----------------------------------------- plot -------------------------------------------------#

    for i in range(ntrails):
        if ctf.comm().rank() == 0:
            plt.plot(blockCGtimeList[i], blockCGerrList[i], label = "trail %d" %(i) )
            plt.legend()
            plt.title("Function tensor(%d*%d*%d), iterative block CG, block size %d, rank %d, sparsity %f" % (I,J,K,block,r,sparsity))
            plt.xlabel("Time[s]")
            plt.ylabel("Training Error Norm")
            plt.savefig('iterative_block_CG.png', bbox_inches='tight')
            

   
    plt.figure()
    for i in range(ntrails):
        if ctf.comm().rank() == 0:
            plt.plot(KressnertimeList[i], KressnererrList[i], label = "trail %d" %(i))
            plt.legend()
            plt.title("Funtion tensor(%d*%d*%d),direct CG (Kressner), rank %d, sparsity %f " % (I,J,K,r,sparsity))
            plt.xlabel("Time[s]")
            plt.ylabel("Training Error Norm")
            plt.savefig('picture.png', bbox_inches='tight')



main()

