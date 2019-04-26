import ctf
T = ctf.tensor((12092,9184,28818), sp=True)
T.read_from_file('nell-2.tns')
assert(T[0,182,606] == 1.0)
