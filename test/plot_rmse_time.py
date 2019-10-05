import matplotlib.pyplot as plt
import matplotlib.lines as mlines

method = []
rmse = []
time = []

def parse_output(file_name):
    num_method = 0
    with open(file_name, 'r') as f:
        for line in f:
            if line[2:9] == 'Initial':
                initial_rmse = float(line.split(',')[1][1:-2])

            if line[2:12] == 'Performing':
                method.append(line.split('seconds')[1][4:-3])
                rmse.append([initial_rmse])
                time.append([0])
                num_method += 1

            if line[2:6] == 'RMSE':
                rmse[num_method-1].append(float(line.split(':')[1][1:-3]))
                time[num_method-1].append(float(line.split(',')[1][1:]))


def plot_each():
    for i in range(len(method)):
        plt.plot(time[i], rmse[i], 'ro')
        plt.plot(time[i], rmse[i])
        plt.title(method[i])
        plt.xlabel('Time (seconds)')
        plt.ylabel('RMSE (average error)')
        plt.show()

def plot_together():
    plt.figure(figsize=(7,4))
    colors = ['r', 'b', 'g', 'brown']
    linestyles = ['--', '-.', ':', '-']
    markers = ['s', 'x', '^', 'd']
    for i in range(len(method))[::-1]:
        plt.plot(time[i], rmse[i], color=colors[i], linestyle=linestyles[i], marker=markers[i], label=method[i])
    plt.title('Tensor Completion using Cyclops on 4096 Cores of Stampede2')
    plt.xlabel('Time (seconds)')
    plt.ylabel('RMSE (average entrywise model error)')
    plt.yscale("log")
    plt.legend(loc='best')
    plt.grid(linestyle='--')
    plt.show()

parse_output('pp.bench.ss.N64.func.out')
#plot_each()
plot_together()
