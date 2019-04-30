import ctf

glob_comm = ctf.comm()

def creat_perfect_tensor(I, J, K, r, sparsity):
    U = ctf.random.random((I, r))
    V = ctf.random.random((J, r))
    W = ctf.random.random((K, r))
    T = ctf.tensor((I, J, K), sp=True)
    T.i("ijk") << U.i("ir") * V.i("jr") * W.i("kr")
    T.sample(sparsity)
    return T