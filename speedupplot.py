# Plots serial vs. OpenMP vs. CUDA timings
# (C) Tanya Wang, Sonia Reilly, Nolan Reilly, May 2022

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
import numpy as np

# read in 'serialtime.txt'
# contains a column of N and a column of time, tab separated
with open('serialtime.txt') as f:
    lines = f.readlines()

# convert strings to lists
NN = []
serialtt = []
for line in lines:
    linearr = line.split('\t')
    NN.append(int(linearr[0]))
    serialtt.append(float(linearr[1]))

serial = np.array(serialtt)

# read in 'omptime.txt'
# contains a column of N and a column of time, tab separated
with open('omptime.txt') as f:
    lines = f.readlines()

# convert strings to lists
NN = []
omptt = []
for line in lines:
    linearr = line.split('\t')
    NN.append(int(linearr[0]))
    omptt.append(float(linearr[1]))

omp = np.array(omptt)

# read in 'cudatime.txt'
# contains a column of N and a column of time, tab separated
with open('cudatime.txt') as f:
    lines = f.readlines()

# convert strings to lists
NN = []
cudatt = []
for line in lines:
    linearr = line.split('\t')
    NN.append(int(linearr[0]))
    cudatt.append(float(linearr[1]))

cuda = np.array(cudatt)

fig, ax = plt.subplots()
ax.set_xscale('log', base=2)
ax.set_yscale('log', base=2)
ax.set_xlabel('N')
ax.set_ylabel('time (s)')
ax.set_title('Multigrid Runtimes')

ax.plot(NN,serial,label='serial')
ax.plot(NN,omp,label='OpenMP')
ax.plot(NN,cuda,label='CUDA')
ax.legend()
plt.savefig('runtimes.pdf')
