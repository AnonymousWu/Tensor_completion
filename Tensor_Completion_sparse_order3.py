
import ctf,time,random
import numpy as np
import numpy.linalg as la
from ctf import random as crandom
glob_comm = ctf.comm()
from scipy.sparse.linalg import lsqr as lsqr

# In[6]:


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


# In[7]:


def normalize(Z,r):
    norms = ctf.tensor(r)
    norms.i("u") << Z.i("pu")*Z.i("pu")
    norms = 1./norms**.5
    X = ctf.tensor(copy=Z)
    Z.set_zero()
    Z.i("pu") << X.i("pu")*norms.i("u")
    return 1./norms


# In[8]:


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
    omegactf = ((T > 0)*ctf.astensor(1.))
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


# In[9]:


def LS_SVD(Z,factor,r,Tbar,regParam,idx):
    [U_,S_,VT_] = ctf.svd(Z)
    S_ = S_/(S_*S_ + regParam*ctf.ones(S_.shape))
    #print(Tbar)
    factor.set_zero()
    factor.i("r") << VT_.i("kr")*S_.i("k")*U_.i("tk")*Tbar.i("t")
    #print( VT_.to_nparray().T.shape,S_.to_nparray().shape,U_.to_nparray().T.shape,Tbar.to_nparray().shape)
    #factor = VT_.to_nparray().T@S_.to_nparray()@U_.to_nparray().T@Tbar.to_nparray()
    #if Z.shape[0] == 1:
    #    print("Here", factor,VT_,S_,U_,Tbar)
    
    return factor    

def updateU_SVD(T,U,V,W,regParam,omega,I,J,K,r):

    M1 = ctf.tensor((J,K,r))
    M1.i("jku") << V.i("ju")*W.i("ku")
    
    for i in range(I):
        num_nonzero, dense_omega = getDenseOmega(T,U,V,W,regParam,omega,I,J,K,r,i,"i")

        Z = ctf.tensor((num_nonzero,r))
        Z.i("tr") << dense_omega.i("jkt")*M1.i("jkr")
        
        Tbar = ctf.tensor((num_nonzero))
        Tbar.i("t") << dense_omega.i("jkt") *T[i,:,:].i("jk")
        
        #print("in",U[i])
        U[i,:].set_zero()
        U[i,:] = LS_SVD(Z,U[i,:],r,Tbar,regParam,i)
        #print("out",U[i])   
 
    #print(U)
    #print(normalize(U,r))
    #U *= normalize(U,r)
     
    return U
    
    
def updateV_SVD(T,U,V,W,regParam,omega,I,J,K,r):
    '''Update V matrix by using the formula'''
    
    M2 = ctf.tensor((I,K,r))
    M2.i("iku") << U.i("iu")*W.i("ku")
    
    for j in range(J):
        num_nonzero, dense_omega = getDenseOmega(T,U,V,W,regParam,omega,I,J,K,r,j,"j")
        Z = ctf.tensor((num_nonzero,r))
        Z.i("tr") << dense_omega.i("ikt")*M2.i("ikr")
        
        Tbar = ctf.tensor((num_nonzero))
        Tbar.i("t") << dense_omega.i("ikt") *T[:,j,:].i("ik")
        
        V[j,:].set_zero()
        V[j,:] = LS_SVD(Z,V[j,:],r,Tbar,regParam,j)
        
    #V *= normalize(V,r)
    
    return V  

def updateW_SVD(T,U,V,W,regParam,omega,I,J,K,r):
    '''Update V matrix by using the formula'''
    
    M3 = ctf.tensor((I,J,r))
    M3.i("iju") << U.i("iu")*V.i("ju")
    
    for k in range(K):
        num_nonzero, dense_omega = getDenseOmega(T,U,V,W,regParam,omega,I,J,K,r,k,"k")
        Z = ctf.tensor((num_nonzero,r))
        Z.i("tr") << dense_omega.i("ijt")*M3.i("ijr")
        
        Tbar = ctf.tensor((num_nonzero))
        Tbar.i("t") << dense_omega.i("ijt") *T[:,:,k].i("ij")
        
        W[k,:].set_zero()
        W[k,:] = LS_SVD(Z,W[k,:],r,Tbar,regParam,k)
       
    #W *= normalize(W,r)
    
    return W


def LS_CG(Ax0,b,Z,x0,r,regParam):
    rk = b - Ax0
    sk = rk
    xk = x0
    for i in range(sk.shape[0]): # how many iterations?
        Ask = ctf.tensor(r)
        Ask.i("i") << Z.i("ti") * Z.i("tj") * sk.i("j")  # A @ sk
        Ask += regParam*sk
        rnorm = ctf.dot(rk,rk)
        #print("rnorm",rnorm.to_nparray())
        if rnorm.to_nparray() < 1.e-16:
            break
        alpha = rnorm/ctf.dot(sk, Ask)
        xk1 = xk + alpha * sk
        rk1 = rk - alpha * Ask
        beta = ctf.dot(rk1,rk1)/rnorm
        sk1 = rk1 + beta*sk
        rk = rk1
        xk = xk1
        sk = sk1
        #print("rk",ctf.vecnorm(rk))
    return xk


def CG(Z,Tbar,r,regParam):
    x0 = ctf.random.random(r)
    Ax0 = ctf.tensor((r))
    Ax0.i("i") << Z.i("ti") * Z.i("tj") * x0.i("j")  # LHS; ATA using matrix-vector multiplication
    Ax0 += regParam*x0
    b = ctf.dot(Z.transpose(),Tbar)                  # RHS; ATb
    return LS_CG(Ax0,b,Z,x0,r,regParam)


def updateU_CG(T,U,V,W,regParam,omega,I,J,K,r):
    
    M1 = ctf.tensor((J,K,r))
    M1.i("jku") << V.i("ju")*W.i("ku")
    
    for i in range(I):
        #num_nonzero, dense_omega = getDenseOmega(T,U,V,W,regParam,omega,I,J,K,r,i,"i")

        #Z = ctf.tensor((num_nonzero,r))
        #Z.i("tr") << dense_omega.i("jkt")*M1.i("jkr")
        
        #Tbar = ctf.tensor((num_nonzero))
        #Tbar.i("t") << dense_omega.i("jkt") *T[i,:,:].i("jk")
        
        
        #U[i,:].set_zero()
        #U[i,:] = CG(Z,Tbar,r,regParam)
        #U[i,:] = la.lstsq(ctf.to_nparray(Z), ctf.to_nparray(Tbar))[0]

        
        Z = M1.reshape((J*K,r))
        Tbar = T[i,:,:].reshape((J*K))
       
        U[i,:].set_zero()
        U[i,:] = CG(Z,Tbar,r,regParam)
    #print(U)
    #print(normalize(U,r))
    #U *= normalize(U,r)
     
    return U
    
    
def updateV_CG(T,U,V,W,regParam,omega,I,J,K,r):
    '''Update V matrix by using the formula'''
    
    M2 = ctf.tensor((I,K,r))
    M2.i("iku") << U.i("iu")*W.i("ku")
    
    for j in range(J):
        num_nonzero, dense_omega = getDenseOmega(T,U,V,W,regParam,omega,I,J,K,r,j,"j")
        Z = ctf.tensor((num_nonzero,r))
        Z.i("tr") << dense_omega.i("ikt")*M2.i("ikr")
        
        Tbar = ctf.tensor((num_nonzero))
        Tbar.i("t") << dense_omega.i("ikt") *T[:,j,:].i("ik")
        
        V[j,:].set_zero()
        V[j,:] = CG(Z,Tbar,r,regParam)
        #V[j,:] = la.lstsq(ctf.to_nparray(Z), ctf.to_nparray(Tbar))[0]

        # Z = M2.reshape((I*K,r))
        # Tbar = T[:,j,:].reshape((I*K))
        # V[j,:].set_zero()
        # V[j,:] = CG(Z,Tbar,r)


    #V *= normalize(V,r)
    
    return V  

def updateW_CG(T,U,V,W,regParam,omega,I,J,K,r):
    '''Update V matrix by using the formula'''
    
    M3 = ctf.tensor((I,J,r))
    M3.i("iju") << U.i("iu")*V.i("ju")
    
    for k in range(K):
        num_nonzero, dense_omega = getDenseOmega(T,U,V,W,regParam,omega,I,J,K,r,k,"k")
        Z = ctf.tensor((num_nonzero,r))
        Z.i("tr") << dense_omega.i("ijt")*M3.i("ijr")
        
        Tbar = ctf.tensor((num_nonzero))
        Tbar.i("t") << dense_omega.i("ijt") *T[:,:,k].i("ij")
   
        W[k,:].set_zero()
        W[k,:] = CG(Z,Tbar,r,regParam)
        #W[k,:] = la.lstsq(ctf.to_nparray(Z), ctf.to_nparray(Tbar))[0]

        # Z = M3.reshape((J*K,r))
        # Tbar = T[:,:,k].reshape((J*K))
        # W[k,:].set_zero()
        # W[k,:] = CG(Z,Tbar,r)
    #print(W,normalize(W,r))
    #W *= normalize(W,r)
    
    return W


# In[18]:


def getALSctf(T,U,V,W,regParam,omega,I,J,K,r):
    '''
    Same thing as above, but CTF
    '''
    it = 0
    E = ctf.tensor((I,J,K))
    E.i("ijk") << T.i("ijk") - omega.i("ijk")*U.i("iu")*V.i("ju")*W.i("ku")
    curr_err_norm = ctf.vecnorm(E) + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W))*regParam
    
    while True:
        
        #U = updateU_SVD(T,U,V,W,regParam,omega,I,J,K,r)
        #V = updateV_SVD(T,U,V,W,regParam,omega,I,J,K,r) 
        #W = updateW_SVD(T,U,V,W,regParam,omega,I,J,K,r)
        
        U = updateU_CG(T,U,V,W,regParam,omega,I,J,K,r)
        V = updateV_CG(T,U,V,W,regParam,omega,I,J,K,r) 
        W = updateW_CG(T,U,V,W,regParam,omega,I,J,K,r)
        
        E.set_zero()
        E.i("ijk") << T.i("ijk") - omega.i("ijk")*U.i("iu")*V.i("ju")*W.i("ku")
        next_err_norm = ctf.vecnorm(E) + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W))*regParam
            
        print(curr_err_norm, next_err_norm)
        
        if abs(curr_err_norm - next_err_norm) < .001 or it > 20:
            break
        #print(next_err_norm/curr_err_norm)
        curr_err_norm = next_err_norm
        it += 1
    
    print("Number of iterations: ", it)
    return U,V,W


# In[19]:


def main():
    
    #ut = UnitTests()
    #ut.runAllTests()

    I = random.randint(6,6)
    J = random.randint(6,6)
    K = random.randint(6,6)
    #I = 30
    #J = 30
    #K = 30
    r = 2 
    sparsity = .1
    regParam = 0.1
        
    ctf.random.seed(42)
    U = ctf.random.random((I,r))
    V= ctf.random.random((J,r))
    W= ctf.random.random((K,r))
    
    # 3rd-order tensor
    T = ctf.tensor((I,J,K),sp=True)
    T.fill_sp_random(0,1,sparsity)
 
    #T.i("ijk") << T1.i("ijk")
    #print(T.sp)
    #print(T)
    omega = updateOmega(T,I,J,K)
    
    U = ctf.random.random((I,r))
    V= ctf.random.random((J,r))
    W= ctf.random.random((K,r))
    
    t = time.time()
    
    getALSctf(T,U,V,W,regParam,omega,I,J,K,r)
    
    print("ALS costs time = ",np.round_(time.time()- t,4))    


# In[20]:


main()



