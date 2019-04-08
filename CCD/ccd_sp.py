import ctf
from ctf import random
import matplotlib.pyplot as plt
import time
glob_comm = ctf.comm()
import numpy as np

def function_tensor(I, J, K, sparsity):
    # N = 5
    # n = 51
    # L = 100
    # nsample = 10*N*n*L #10nNL = 255000

    T = ctf.tensor((I, J, K))
    T2 = ctf.tensor((I, J, K))

    T.fill_sp_random(1, 1, sparsity)
    # T = ctf.exp(-1 * ctf.power(ctf.power(T,2),0.5))  # x = exp(-sqrt(x^2))

    sizes = [I, J, K]
    index = ["i", "j", "k"]

    for i in range(3):
        n = sizes[i]
        v = np.linspace(-1, 1, n)
        # v = np.arange(1,n+1)
        v = ctf.astensor(v ** 2)

        v2 = ctf.tensor(n)
        v2 = v
        T2.i("ijk") << T.i("ijk") * v2.i(index[i])

    T2 = ctf.power(T2, 0.5)
    T2 = (-1.0) * T2

    # T2 = ctf.exp(T2)

    return T2

def getOmega(T):
    if not T.sp:
        omegactf = ((T != 0)*ctf.astensor(1.))
    else:
        omegactf = T / T
        assert(omegactf.sp)
    
    return omegactf

def get_objective(T,U,V,W,I,J,K,omega,regParam):
	L = ctf.tensor((I,J,K))
	t0 = time.time()
	L.i("ijk") << T.i("ijk") - ctf.TTTP(omega, [U,V,W]).i("ijk")
	t1 = time.time()
	objective = ctf.vecnorm(L) + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W)) * regParam
	t2 = time.time()
	if glob_comm.rank() == 0:
		print('generate L takes {}'.format(t1 - t0))
		print('calc objective takes {}'.format(t2 - t1))
	return objective

def main():

	I = 100
	J = 100
	K = 100

	r = 30

	sparsity = .1
	regParam = .1

	ctf.random.seed(42)

	# # 3rd-order tensor
	# T = ctf.tensor((I,J,K))
	# # T.read_from_file('tensor.txt')
	# T.fill_sp_random(0,1,sparsity)
	T = function_tensor(I,J,K,sparsity)

	t0 = time.time()
	omega = getOmega(T)
	
	if glob_comm.rank() == 0:
		print('getOmega takes {}'.format(time.time() - t0))


	U = ctf.random.random((I, r))
	V = ctf.random.random((J, r))
	W = ctf.random.random((K, r))

	# print(T)
	# T.write_to_file('tensor_out.txt')
	# assert(T.sp == 1)

	# exit(0)
	# print(U)
	# print(V)
	# print(W)

	ite = 0
	objectives = []

	t_before_loop = time.time()

	while True:

		t0 = time.time()
		R = ctf.copy(T)
		t1 = time.time()

		# R -= ctf.einsum('ijk, ir, jr, kr -> ijk', omega, U, V, W)
		R -= ctf.TTTP(omega, [U,V,W])
		t2 = time.time()
		# R += ctf.einsum('ijk, i, j, k -> ijk', omega, U[:,0], V[:,0], W[:,0])
		R += ctf.TTTP(omega, [U[:,0], V[:,0], W[:,0]])
		t3 = time.time()

		# print(R)
		# exit(0)

		t4 = time.time()
		objective = get_objective(T,U,V,W,I,J,K,omega,regParam)
		if glob_comm.rank() == 0:
			print('ctf.copy() takes {}'.format(t1-t0))
			print('ctf.TTTP() takes {}'.format(t2 - t1))
			print('ctf.TTTP() takes {}'.format(t3 - t2))
			print('get_objective takes {}'.format(time.time()-t4))
			print('Objective: {}'.format(objective))


		objectives.append(objective)

		for f in range(r):
			
			# update U[:,f]
			if glob_comm.rank() == 0:
				print('updating U[:,{}]'.format(f))

			t0 = time.time()
			alphas = ctf.einsum('ijk, j, k -> i', R, V[:,f], W[:,f])
			t1 = time.time()
			betas = ctf.einsum('ijk, j, j, k, k -> i', omega, V[:,f], V[:,f], W[:,f], W[:,f])
			t2 = time.time()
			
			U[:,f] = alphas / (regParam + betas)

			objective = get_objective(T,U,V,W,I,J,K,omega,regParam)
			if glob_comm.rank() == 0:
				print('Objective: {}'.format(objective))
				print('ctf.einsum() takes {}'.format(t1-t0))
				print('ctf.einsum() takes {}'.format(t2-t1))

			objectives.append(objective)


			# update V[:,f]
			if glob_comm.rank() == 0:
				print('updating V[:,{}]'.format(f))
			alphas = ctf.einsum('ijk, i, k -> j', R, U[:,f], W[:,f])
			betas = ctf.einsum('ijk, i, i, k, k -> j', omega, U[:,f], U[:,f], W[:,f], W[:,f])
			
			V[:,f] = alphas / (regParam + betas)

			objective = get_objective(T,U,V,W,I,J,K,omega,regParam)
			if glob_comm.rank() == 0:
				print('Objective: {}'.format(objective))
			objectives.append(objective)

			# exit(0)


			# update W[:,f]
			if glob_comm.rank() == 0:
				print('updating W[:,{}]'.format(f))
			alphas = ctf.einsum('ijk, i, j -> k', R, U[:,f], V[:,f])
			betas = ctf.einsum('ijk, i, i, j, j -> k', omega, U[:,f], U[:,f], V[:,f], V[:,f])
			
			W[:,f] = alphas / (regParam + betas)

			objective = get_objective(T,U,V,W,I,J,K,omega,regParam)
			if glob_comm.rank() == 0:
				print('Objective: {}'.format(objective))
			objectives.append(objective)

			# exit(0)


			# t0 = time.time()
			# R -= ctf.einsum('ijk, i, j, k -> ijk', omega, U[:,f], V[:,f], W[:,f])
			R -= ctf.TTTP(omega, [U[:,f], V[:,f], W[:,f]])
			# R += ctf.einsum('ijk, i, j, k -> ijk', omega, U[:,f+1], V[:,f+1], W[:,f+1])
			R += ctf.TTTP(omega, [U[:,f+1], V[:,f+1], W[:,f+1]])
			# print(time.time() - t0)

			# print(R)
			# exit(0)
		
		ite += 1

		if ite == 1:
			break

	if glob_comm.rank() == 0:
		print('Time/Iteration: {}'.format((time.time() - t_before_loop)/1))

	# plt.plot(objectives)
	# plt.yscale('log')
	# plt.show()
	# print(len(objectives))

if __name__ == '__main__':
	main()
