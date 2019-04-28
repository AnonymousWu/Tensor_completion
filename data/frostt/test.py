import ctf
import reader
import sys
glob_comm = ctf.comm()

file_name = sys.argv[1]
I = int(sys.argv[2])
J = int(sys.argv[3])
K = int(sys.argv[4])

if glob_comm.rank() == 0:
    print("I is",I,"J is",J,"K is",K)

T = reader.read_from_frostt(file_name, I, J, K)
if file_name == 'nell-2':
    assert(T[0,182,606] == 1.0)
elif file_name == 'amazon-reviews':
    assert(T[0,305245,32024] == 1.0)
elif file_name == 'patents':
    assert(T[0,0,1] == 1.099000)
elif file_name == 'reddit-2015':
    assert(T[0,52341,80007] == 1)
