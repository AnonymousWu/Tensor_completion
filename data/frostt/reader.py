import gzip
import shutil
import ctf
glob_comm = ctf.comm()
import os

modify = False

def read_from_frostt(file_name, I, J, K):

	unzipped_file_name = file_name + '.tns'
	exists = os.path.isfile(unzipped_file_name)

	if not exists:
		if glob_comm.rank() == 0:
			print('Creating ' + unzipped_file_name)
		with gzip.open(file_name + '.tns.gz', 'r') as f_in:
			with open(unzipped_file_name, 'w') as f_out:
				shutil.copyfileobj(f_in, f_out)
	
	T_start = ctf.tensor((I+1, J+1, K+1), sp=True)
	if glob_comm.rank() == 0:
		print('T_start initialized')
	T_start.read_from_file(unzipped_file_name)
	if glob_comm.rank() == 0:
		print('T_start read in')
	T = ctf.tensor((I,J,K), sp=True)
	if glob_comm.rank() == 0:
		print('T initialized')
	T[:,:,:] = T_start[1:,1:,1:]
	if glob_comm.rank() == 0:
		print('T filled')

	if modify:
		T.write_to_file(unzipped_file_name)

	return T
