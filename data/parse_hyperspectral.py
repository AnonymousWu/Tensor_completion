from scipy.io import loadmat

out_name = 'hyperspectral.txt'

image_arr = loadmat('ref_ribeira1bbb_reg1.mat')['reflectances']
I,J,K = image_arr.shape
print(I,J,K)

with open(out_name, 'w') as f:
	for i in range(I):
		for j in range(J):
			for k in range(K):
				f.write('{} {} {} {}\n'.format(i,j,k,image_arr[i,j,k]))

print('Finished')
