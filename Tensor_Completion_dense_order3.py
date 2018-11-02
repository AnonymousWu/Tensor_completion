
# coding: utf-8

# ### Dense Order-3 Tensor Completion 
# #### Author: Xiaoxiao Wu

# In[1]:

import ctf,time,random
import numpy as np
import numpy.linalg as la
from ctf import random as crandom
glob_comm = ctf.comm()


# #### Test Cases

# In[2]:

class UnitTests:
        
    def test_3d_purturb1(self):
        
        I = random.randint(3,5)  #random dimensions
        J = random.randint(3,5)
        K = random.randint(3,5)
        r = 2     
        regParam = 0
        sparsity = .2
        
        ctf.random.seed(10)
        # generate factor matrices
        U = ctf.random.random((I,r))
        V= ctf.random.random((J,r))
        W= ctf.random.random((K,r))
        
        T = ctf.tensor((I,J,K))
        T.i("ijk") << U.i("iu")*V.i("ju")*W.i("ku")
        
        omega = updateOmega(I,J,K,sparsity)
        
        # purturb the first factor matrix
        U += crandom.random((I,r))*.01
        # call updateU function
        nU = updateU(T,U,V,W,regParam,omega,I,J,K,r)
        
        nT = ctf.tensor((I,J,K))
        nT.i("ijk") << nU.i("iu")*V.i("ju")*W.i("ku")
    
        assert(ctf.all(ctf.abs(nT - T < 1e-10)))
        print("passed test: test_3d_purturb1")
        
    def test_3d_purturb2(self):
        I = random.randint(3,5)  #random dimensions
        J = random.randint(3,5)
        K = random.randint(3,5)
        r = 2     
        regParam = 0
        sparsity = .2
        
        ctf.random.seed(10)
        # generate factor matrices
        U = ctf.random.random((I,r))
        V= ctf.random.random((J,r))
        W= ctf.random.random((K,r))
        
        T = ctf.tensor((I,J,K))
        T.i("ijk") << U.i("iu")*V.i("ju")*W.i("ku")
        
        omega = updateOmega(I,J,K,sparsity)
        
        # purturb the first and second factor matrix
        #U += crandom.random((I,r))*.01
        V += crandom.random((J,r))*.01
        # call updateU function
        #nU = updateU(T,U,V,W,regParam,omega,I,J,K,r)
        nV = updateV(T,U,V,W,regParam,omega,I,J,K,r)
        
        nT = ctf.tensor((I,J,K))
        nT.i("ijk") << U.i("iu")*nV.i("ju")*W.i("ku")
    
        assert(ctf.all(ctf.abs(nT - T < 1e-10)))
        print("passed test: test_3d_purturb2")
        
        
    def test_3d_purturb3(self):
        I = random.randint(3,5)  #random dimensions
        J = random.randint(3,5)
        K = random.randint(3,5)
        r = 2     
        regParam = 0
        sparsity = .2
        
        ctf.random.seed(10)
        # generate factor matrices
        U = ctf.random.random((I,r))
        V= ctf.random.random((J,r))
        W= ctf.random.random((K,r))
        
        T = ctf.tensor((I,J,K))
        T.i("ijk") << U.i("iu")*V.i("ju")*W.i("ku")
        
        omega = updateOmega(I,J,K,sparsity)
        
        # purturb the first and second factor matrix
        #U += crandom.random((I,r))*.01
        W += crandom.random((K,r))*.01
        # call updateU function
        nW = updateW(T,U,V,W,regParam,omega,I,J,K,r)
        #nU = updateU(T,U,V,W,regParam,omega,I,J,K,r)
        #nV = updateV(T,U,V,W,regParam,omega,I,J,K,r)
        
        nT = ctf.tensor((I,J,K))
        nT.i("ijk") << U.i("iu")*V.i("ju")*nW.i("ku")
    
        assert(ctf.all(ctf.abs(nT - T < 1e-10)))
        print("passed test: test_3d_purturb3")
        

        
    def runAllTests(self):
        self.test_3d_purturb1()
        self.test_3d_purturb2()
        self.test_3d_purturb3()


# In[3]:

def normalize(Z,r):
    norms = ctf.tensor(r)
    norms.i("u") << Z.i("pu")*Z.i("pu")
    norms = 1./norms**.5
    X = ctf.tensor(copy=Z)
    Z.set_zero()
    Z.i("pu") << X.i("pu")*norms.i("u")
    return 1./norms

def updateOmega(I,J,K,sparsity):
    '''
    Gets a random subset of rows for each U,V,W iteration
    '''
    Actf = ctf.tensor((I,J,K),sp=True)
    Actf.fill_sp_random(0,1,sparsity)
    omegactf = ((Actf > 0)*ctf.astensor(1.))
    return omegactf


# In[4]:

def updateU(T,U,V,W,regParam,omega,I,J,K,r):
    '''Update U matrix by using the formula'''
    
    M1 = ctf.tensor((J,K,r))
    M1.i("jku") << V.i("ju")*W.i("ku")
    [U_,S_,V_] = ctf.svd(M1.reshape((J*K,r)))
    #S_ = 1./S_
    S_ = S_/(S_*S_ + regParam*ctf.ones(r))
    U.set_zero()
    U.i("iu") << V_.i("vu")*S_.i("v")*U_.reshape((J,K,r)).i("jkv")*T.i("ijk")
    U *= normalize(U,r)
    
    return U
    
    
def updateV(T,U,V,W,regParam,omega,I,J,K,r):
    '''Update V matrix by using the formula'''
    
    M2 = ctf.tensor((I,K,r))
    M2.i("iku") << U.i("iu")*W.i("ku")
    [U_,S_,V_] = ctf.svd(M2.reshape((I*K,r)))
    #S_ = 1./S_
    S_ = S_/(S_*S_ + regParam*ctf.ones(r))
    V.set_zero()
    V.i("ju") << V_.i("vu")*S_.i("v")*U_.reshape((I,K,r)).i("ikv")*T.i("ijk")
    #normalize(V,r)
    V *= normalize(V,r)
    
    return V  

def updateW(T,U,V,W,regParam,omega,I,J,K,r):
    '''Update V matrix by using the formula'''
    
    M3 = ctf.tensor((I,J,r))
    M3.i("iju") << U.i("iu")*V.i("ju")
    [U_,S_,V_] = ctf.svd(M3.reshape((I*J,r)))
    #S_ = 1./S_
    S_ = S_/(S_*S_ + regParam*ctf.ones(r))
    W.set_zero()
    W.i("ku") << V_.i("vu")*S_.i("v")*U_.reshape((I,J,r)).i("ijv")*T.i("ijk")
    W *= normalize(W,r)
    
    return W


# In[5]:

def getALSCtf(T,U,V,W,regParam,omega,I,J,K,r):
    '''
    Same thing as above, but CTF
    '''
    it = 0
    E = ctf.tensor((I,J,K))
    E.i("ijk") << T.i("ijk") - U.i("iu")*V.i("ju")*W.i("ku")
    curr_err_norm = ctf.vecnorm(E) + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W))*regParam
    
    while True:
        U = updateU(T,U,V,W,regParam,omega,I,J,K,r)
        V = updateV(T,U,V,W,regParam,omega,I,J,K,r) 
        W = updateW(T,U,V,W,regParam,omega,I,J,K,r)
        E.set_zero()
        E.i("ijk") << T.i("ijk") - U.i("iu")*V.i("ju")*W.i("ku")
        next_err_norm = ctf.vecnorm(E) + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W))*regParam
        
        if abs(curr_err_norm - next_err_norm) < .001 or it > 100:
            break
            
        print(curr_err_norm, next_err_norm)
        curr_err_norm = next_err_norm
        it += 1
    
    print("Number of iterations: ", it)
    return U,V,W


# In[6]:

def main():
    
    ut = UnitTests()
    ut.runAllTests()

    I = random.randint(30,50)
    J = random.randint(30,50)
    K = random.randint(30,50)
    r = 2 
    sparsity = .2
    regParam = 10
        
    ctf.random.seed(42)
    U = ctf.random.random((I,r))
    V= ctf.random.random((J,r))
    W= ctf.random.random((K,r))
        
    # 3rd-order tensor
    T = ctf.tensor((I,J,K))
    #T.fill_random(0,1)
    T.i("ijk") << U.i("iu")*V.i("ju")*W.i("ku")
    U = ctf.random.random((I,r))
    V= ctf.random.random((J,r))
    W= ctf.random.random((K,r))
    
    omega = updateOmega(I,J,K,sparsity)
    
    t = time.time()
    
    getALSCtf(T,U,V,W,regParam,omega,I,J,K,r)
    
    print("ALS costs time = ",np.round_(time.time()- t,4))    


# In[7]:

main()

