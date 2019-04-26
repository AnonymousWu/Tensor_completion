import gzip
import shutil
import ctf

modify = False

def read_from_frostt(file_name, I, J, K):

	unzipped_file_name = file_name + '.tns'

	with gzip.open(file_name + '.tns.gz', 'r') as f_in:
		with open(unzipped_file_name, 'w') as f_out:
			shutil.copyfileobj(f_in, f_out)

	T_start = ctf.tensor((I+1, J+1, K+1), sp=True)
	T_start.read_from_file(unzipped_file_name)
	T = ctf.tensor((I,J,K), sp=True)
	T[:,:,:] = T_start[1:,1:,1:]

	if modify:
		T.write_to_file(unzipped_file_name)

	return T