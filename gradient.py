import ctf,time,random
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from ctf import random as crandom
glob_comm = ctf.comm()

def getOmega(T):
    omegactf = ((T > 0)*ctf.astensor(1.))
    return omegactf

def dense_updateU(T,U,V,W,Lambda,I,J,K,r,R, stepSize):
    H = ctf.tensor((J,K,r))
    H.i("jku") << V.i("ju")*W.i("ku")
    U.i("ir") << - stepSize*(2*Lambda*U.i("ir") - R.i("ijk")*H.i("jkr"))
    return U

def dense_updateV(T,U,V,W,Lambda,I,J,K,r,R,stepSize):
    H = ctf.tensor((I,K,r))
    H.i("iku") << U.i("iu")*W.i("ku")
    V.i("jr") << - stepSize*(2*Lambda*V.i("jr") - R.i("ijk")*H.i("ikr"))
    return V

def dense_updateW(T,U,V,W,Lambda,I,J,K,r,R,stepSize):
    H = ctf.tensor((I,J,r))
    H.i("iju") << U.i("iu")*V.i("ju")
    W.i("kr") << - stepSize*(2*Lambda*W.i("kr") - R.i("ijk")*H.i("ijr"))
    return W

def sparse_updateU(T,U,V,W,Lambda,omega,I,J,K,r,R, stepSize):
    H = ctf.tensor((J,K,r))
    H.i("jku") << V.i("ju")*W.i("ku")
    U.i("ir") << - stepSize*(2*Lambda*U.i("ir") - omega.i("ijk")*R.i("ijk")*H.i("jkr"))
    return U

def sparse_updateV(T,U,V,W,Lambda,omega,I,J,K,r,R,stepSize):
    H = ctf.tensor((I,K,r))
    H.i("iku") << U.i("iu")*W.i("ku")
    V.i("jr") << - stepSize*(2*Lambda*V.i("jr") - omega.i("ijk")*R.i("ijk")*H.i("ikr"))
    return V

def sparse_updateW(T,U,V,W,Lambda,omega,I,J,K,r,R,stepSize):
    H = ctf.tensor((I,J,r))
    H.i("iju") << U.i("iu")*V.i("ju")
    W.i("kr") << - stepSize*(2*Lambda*W.i("kr") - omega.i("ijk")*R.i("ijk")*H.i("ijr"))
    return W

def dense_GD(T,U,V,W,Lambda,I,J,K,r,stepSize):
    iteration_count = 0
    E = ctf.tensor((I,J,K))
    E.i("ijk") << T.i("ijk") - U.i("iu")*V.i("ju")*W.i("ku")
    curr_err_norm = ctf.vecnorm(E) + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W))*Lambda
    
    while True:
        U = dense_updateU(T,U,V,W,Lambda,I,J,K,r,E,stepSize)
        V = dense_updateV(T,U,V,W,Lambda,I,J,K,r,E,stepSize)
        W = dense_updateW(T,U,V,W,Lambda,I,J,K,r,E,stepSize)
        E.set_zero()
        E.i("ijk") << T.i("ijk") - U.i("iu")*V.i("ju")*W.i("ku")
        next_err_norm = ctf.vecnorm(E) + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W))*Lambda
        
        if abs(curr_err_norm - next_err_norm) < .001 or iteration_count > 100:
            break
            
        print(curr_err_norm, next_err_norm)
        curr_err_norm = next_err_norm
        iteration_count += 1
    
    print("Number of iterations: ", iteration_count)
    return U,V,W


def sparse_GD(T, U, V, W, Lambda, omega, I, J, K, r, stepSize):
    iteration_count = 0
    E = ctf.tensor((I, J, K))
    E.i("ijk") << T.i("ijk") - omega.i("ijk") * U.i("iu") * V.i("ju") * W.i("ku")
    curr_err_norm = ctf.vecnorm(E) + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W)) * Lambda
    norm = [curr_err_norm]
    while True:
        U = sparse_updateU(T, U, V, W, Lambda, omega, I, J, K, r, E, stepSize)
        E.set_zero()
        E.i("ijk") << T.i("ijk") - omega.i("ijk") * U.i("iu") * V.i("ju") * W.i("ku")
        V = sparse_updateV(T, U, V, W, Lambda, omega, I, J, K, r, E, stepSize)
        E.set_zero()
        E.i("ijk") << T.i("ijk") - omega.i("ijk") * U.i("iu") * V.i("ju") * W.i("ku")
        W = sparse_updateW(T, U, V, W, Lambda, omega, I, J, K, r, E, stepSize)
        E.set_zero()
        E.i("ijk") << T.i("ijk") - omega.i("ijk") * U.i("iu") * V.i("ju") * W.i("ku")
        next_err_norm = ctf.vecnorm(E) + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W)) * Lambda

        if abs(curr_err_norm - next_err_norm) < .01 or iteration_count > 100:
            break

        print(curr_err_norm, next_err_norm, ctf.vecnorm(E))
        curr_err_norm = next_err_norm
        if iteration_count % 5 == 0:
            norm.append(curr_err_norm)
        iteration_count += 1

    print("Number of iterations: ", iteration_count)
    return norm

def sparse_SGD(T,U,V,W,Lambda,omega,I,J,K,r,stepSize,sample_rate):
    iteration_count = 0
    E = ctf.tensor((I,J,K))
    R = ctf.tensor((I,J,K))
    R.i("ijk") << T.i("ijk") - omega.i("ijk")*U.i("iu")*V.i("ju")*W.i("ku")
    sampled_T = ctf.tensor((I,J,K))
    curr_err_norm = ctf.vecnorm(R) + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W))*Lambda
    norm = [curr_err_norm]
    while True:
        random = ctf.tensor((I,J,K))
        random.fill_random(0, 1)
        random = ((random > sample_rate)*ctf.astensor(1.))
        sampled_T = T * random
        #sampled_T.sample(sample_rate)
        sampled_omega = getOmega(sampled_T)
        E.i("ijk") << sampled_T.i("ijk") - sampled_omega.i("ijk") * U.i("iu") * V.i("ju") * W.i("ku")
        U = sparse_updateU(sampled_T,U,V,W,Lambda,sampled_omega,I,J,K,r,E,stepSize)
        E.set_zero()
        E.i("ijk") << sampled_T.i("ijk") - sampled_omega.i("ijk") * U.i("iu") * V.i("ju") * W.i("ku")
        V = sparse_updateV(sampled_T,U,V,W,Lambda,sampled_omega,I,J,K,r,E,stepSize)
        E.set_zero()
        E.i("ijk") << sampled_T.i("ijk") - sampled_omega.i("ijk") * U.i("iu") * V.i("ju") * W.i("ku")
        W = sparse_updateW(sampled_T,U,V,W,Lambda,sampled_omega,I,J,K,r,E,stepSize)
        E.set_zero()
        sampled_T.set_zero()
            
        if iteration_count % 5 == 0:
            R.set_zero()
            R.i("ijk") << T.i("ijk") - omega.i("ijk")*U.i("iu")*V.i("ju")*W.i("ku")
            diff_norm = ctf.vecnorm(R)
            next_err_norm = diff_norm + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W))*Lambda
            print(curr_err_norm, next_err_norm)
            print(diff_norm)

            if abs(curr_err_norm - next_err_norm) < .01 or iteration_count > 100:
                break

            curr_err_norm = next_err_norm
            norm.append(curr_err_norm)

        iteration_count += 1
    
    print("Number of iterations: ", iteration_count)
    return norm


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
        random = ((random > sample_rate) * ctf.astensor(1.))
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

            if abs(curr_err_norm - next_err_norm) < .001 or iteration_count > 100:
                break

            curr_err_norm = next_err_norm

        iteration_count += 1

    print("Number of iterations: ", iteration_count)
    return U, V, W

def main():
    size_lb = 40
    size_ub = 60
    r = 20
    sparsity = .1
    regParam = 0.1
    stepSize = 0.0001
    sample_rate = 0.0001


    I = random.randint(size_lb,size_ub)
    J = random.randint(size_lb,size_ub)
    K = random.randint(size_lb,size_ub)

    target_tensor = ctf.tensor((I, J, K))
    target_tensor.fill_random(0, 1)

    ctf.random.seed(42)
    target_U = ctf.random.random((I,r))
    target_V = ctf.random.random((J,r))
    target_W = ctf.random.random((K,r))

    dense_SGD(target_tensor,target_U,target_V,target_W,regParam,I,J,K,r,stepSize, sample_rate)

    #T = ctf.tensor((I,J,K),sp=True)
    T = ctf.tensor((I,J,K))
    T.i("ijk") << target_U.i("iu") * target_V.i("ju") * target_W.i("ku")
    #T.fill_sp_random(0,1,sparsity)
    rand = ctf.tensor((I,J,K))
    rand.fill_random(0, 1)
    rand = ((rand < sparsity)*ctf.astensor(1.))
    T = T * rand
    omega = getOmega(T)
    
    U = ctf.random.random((I,r))
    V= ctf.random.random((J,r))
    W= ctf.random.random((K,r))
    stepSize = 0.0005
    t = time.time()
    #sparse_GD(T,target_U,target_V,target_W,regParam,omega,I,J,K,r,stepSize)
    plt.plot(sparse_GD(T, U, V, W, regParam, omega, I, J, K, r, stepSize), label = "GD")
    print("GD total time = ", np.round_(time.time() - t, 4))

    U = ctf.random.random((I, r))
    V = ctf.random.random((J, r))
    W = ctf.random.random((K, r))
    t = time.time()
    plt.plot(sparse_SGD(T,U,V,W,regParam,omega,I,J,K,r,stepSize, sample_rate), label = "SGD")
    plt.legend()
    plt.title("tensor dimension: "+str(I)+"," + str(J)+","+str(K) +" rank: " + str(r) + " sparsity: " + str(sparsity))
    plt.show()
    print("SGD total time = ",np.round_(time.time() - t,4))

# TO DO
# Step size picking
# plotting
# compare to als

main()