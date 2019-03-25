import ctf
from ctf import random
import matplotlib.pyplot as plt
import time
glob_comm = ctf.comm()
import numpy as np


def getOmega(T):
    omegactf = ((T > 0)*ctf.astensor(1.))
    return omegactf

def get_objective(T,U,V,W,I,J,K,omega,regParam):
	L = ctf.tensor((I,J,K))
	L.i("ijk") << T.i("ijk") - omega.i("ijk")*U.i("iu")*V.i("ju")*W.i("ku")
	return ctf.vecnorm(L) + (ctf.vecnorm(U) + ctf.vecnorm(V) + ctf.vecnorm(W)) * regParam

def main():

	I = 100
	J = 100
	K = 100

	r = 30

	sparsity = .8
	regParam = .1

	ctf.random.seed(42)

	# 3rd-order tensor
	T = ctf.tensor((I,J,K),sp=True)
	# T.read_from_file('tensor.txt')
	T.fill_sp_random(0,1,sparsity)
	assert(T.sp == 1)

	omega = getOmega(T)


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

	t0 = time.time()

	while True:

		R = ctf.copy(T)
		R -= ctf.einsum('ijk, ir, jr, kr -> ijk', omega, U, V, W)
		R += ctf.einsum('ijk, i, j, k -> ijk', omega, U[:,0], V[:,0], W[:,0])
		
		# print(R)
		# exit(0)

		objective = get_objective(T,U,V,W,I,J,K,omega,regParam)
		if glob_comm.rank() == 0:
			print('Objective: {}'.format(objective))
		objectives.append(objective)

		# exit(0)

		for f in range(r):
			
			# update U[:,f]
			if glob_comm.rank() == 0:
				print('updating U[:,{}]'.format(f))
			alphas = ctf.einsum('ijk, j, k -> i', R, V[:,f], W[:,f])
			betas = ctf.einsum('ijk, j, j, k, k -> i', omega, V[:,f], V[:,f], W[:,f], W[:,f])
			
			U[:,f] = alphas / (regParam + betas)

			objective = get_objective(T,U,V,W,I,J,K,omega,regParam)
			if glob_comm.rank() == 0:
				print('Objective: {}'.format(objective))
			objectives.append(objective)

			# exit(0)


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
			R -= ctf.einsum('ijk, i, j, k -> ijk', omega, U[:,f], V[:,f], W[:,f])
			R += ctf.einsum('ijk, i, j, k -> ijk', omega, U[:,f+1], V[:,f+1], W[:,f+1])
			# print(time.time() - t0)

			# print(R)
			# exit(0)
		
		ite += 1

		if ite == 1:
			break

	if glob_comm.rank() == 0:
		print('Time: {}'.format(time.time() - t0))
	# plt.plot(objectives)
	# plt.yscale('log')
	# plt.show()
	# print(len(objectives))


main()


# import ctf
# from ctf import random
# import matplotlib.pyplot as plt

# def getOmega(T):
#     omegactf = ((T > 0)*ctf.astensor(1.))
#     return omegactf

# def get_objective(T,U,V,I,J,omega,regParam):
# 	L = ctf.tensor((I,J))
# 	L.i("ij") << T.i("ij") - omega.i("ij")*U.i("iu")*V.i("ju")
# 	print("Objective:")
# 	return ctf.vecnorm(L) + (ctf.vecnorm(U) + ctf.vecnorm(V)) * regParam

# def main():
# 	I = 4
# 	J = 5

# 	r = 3

# 	sparsity = .5
# 	regParam = .0001

# 	# 3rd-order tensor
# 	T = ctf.tensor((I,J),sp=True)
# 	T.fill_sp_random(0,1,sparsity)
# 	assert(T.sp == 1)

# 	omega = getOmega(T)

# 	ctf.random.seed(42)

# 	U = ctf.random.random((I, r))
# 	V = ctf.random.random((J, r))

# 	print(T)
# 	print(U)
# 	print(V)

# 	ite = 0
# 	objectives = []

# 	while True:

# 		R = ctf.copy(T)

# 		for l in range(1, r):
# 			for i in range(I):
# 				for j in range(J):
# 					R.write([[i,j]], [R[i][j] - omega[i][j] * U[i][l] * V[j][l]])

# 		print(R)


# 		for f in range(r):
# 			print(get_objective(T,U,V,I,J,omega,regParam))
# 			objectives.append(get_objective(T,U,V,I,J,omega,regParam))

# 			for i in range(I):
# 				print('U({},{})'.format(i, f))
# 				alpha = 0
# 				beta = 0
# 				for j in range(J):
# 					alpha += omega[i][j] * R[i][j] * V[j][f]
# 					beta += omega[i][j] * V[j][f] ** 2

# 				U.write([[i,f]], [alpha / (regParam + beta)])

# 				print(U)
# 				print(get_objective(T,U,V,I,J,omega,regParam))
# 				objectives.append(get_objective(T,U,V,I,J,omega,regParam))


# 			# exit(0)
			
# 			for j in range(J):
# 				print('V({},{})'.format(j, f))
# 				print(V)
# 				alpha = 0
# 				beta = 0
# 				for i in range(I):
					
# 					alpha += omega[i][j] * R[i][j] * U[i][f]
# 					beta += omega[i][j] * U[i][f] ** 2

# 				V.write([[j,f]], [alpha / (regParam + beta)])

# 				print(V)
# 				print(get_objective(T,U,V,I,J,omega,regParam))
# 				objectives.append(get_objective(T,U,V,I,J,omega,regParam))


# 			# exit(0)
			
# 			for i in range(I):
# 				for j in range(J):
# 					R.write([[i,j]], [R[i][j] - omega[i][j] * U[i][f] * V[j][f]])
# 					R.write([[i,j]], [R[i][j] + omega[i][j] * U[i][f+1] * V[j][f+1]])

# 			print(R)
# 			# exit(0)
		
# 		ite += 1

# 		if ite == 3:
# 			break

# 	print(get_objective(T,U,V,I,J,omega,regParam))
# 	plt.plot(objectives)
# 	plt.show()


# main()