import gzip
def read_frosst(file_name):
	with gzip.open(file_name + '.tns.gz', 'rb') as f_in:
		with open(file_name + '.tns', 'w') as f_out:
			file_content = f_in.read().split('\n')
			print(len(file_content))
			for entry in file_content[:-1]:
				i,j,k,v = entry.split(' ')
				f_out.write('{} {} {} {}\n'.format(int(i)-1, int(j)-1, int(k)-1, v))

read_frosst('nell-2')