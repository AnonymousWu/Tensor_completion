import ctf
import reader
import sys

file_name = sys.argv[1]
I = int(sys.argv[2])
J = int(sys.argv[3])
K = int(sys.argv[4])

T = reader.read_from_frostt(file_name, I, J, K)
if file_name == 'nell-2':
	assert(T[0,182,606] == 1.0)
elif file_name == 'nell-1':
	assert(T[0,17350,8011251] == 1.0)
elif file_name == 'amazon-reviews':
	assert(T[0,305245,32024] == 1.0)
