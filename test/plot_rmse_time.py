import matplotlib.pyplot as plt

method = []
rmse = []
time = []

def parse_output(file_name):
	num_method = 0
	with open(file_name, 'r') as f:
        	for line in f:
                	if line[2:12] == 'Performing':
                        	method.append(line.split('seconds')[1][4:-3])
                        	rmse.append([])
                        	time.append([])
                        	num_method += 1
                	if line[2:6] == 'RMSE':
                        	rmse[num_method-1].append(float(line.split(':')[1][1:-3]))
                        	time[num_method-1].append(float(line.split(',')[1][1:]))


def plot_each():
	for i in range(len(method)):
		plt.plot(time[i], rmse[i], 'ro')
		plt.plot(time[i], rmse[i])
		plt.title(method[i])
		plt.xlabel('Time')
		plt.ylabel('RMSE')
		plt.show()

def plot_together():
	for i in range(len(method)):
		plt.plot(time[i], rmse[i], 'ro')
		plt.plot(time[i], rmse[i], label=method[i])
	plt.xlabel('Time')
	plt.ylabel('RMSE')
	plt.legend()
	plt.show()

parse_output('out.txt')
plot_each()
plot_together()
