# Plotting code for the output of multigrid
# Creates a pcolor plot of the solution on the square domain

# (C) Tanya Wang, Sonia Reilly, Nolan Reilly, May 2022

import matplotlib.pyplot as plt
import numpy as np

# read in 'uT.txt'
# contains a column of i, a column of j, and a column of uT, tab separated
with open('uT.txt') as f:
    lines = f.readlines()

# convert strings to lists
ii = []
jj = []
uu = []
for line in lines:
    linearr = line.split('\t')
    ii.append(int(linearr[0]))
    jj.append(int(linearr[1]))
    uu.append(float(linearr[2]))

# convert lists to array uT
n = int(np.sqrt(len(ii))) - 1
uT = np.zeros((n+1,n+1))
for idx in range(len(ii)):
    uT[ii[idx],jj[idx]] = uu[idx]

# read in 'uTcuda.txt'
# contains a column of i, a column of j, and a column of uT, tab separated
with open('uTcuda.txt') as f:
    lines = f.readlines()

# convert strings to lists
ci = []
cj = []
cu = []
for line in lines:
    linearr = line.split('\t')
    ci.append(int(linearr[0]))
    cj.append(int(linearr[1]))
    cu.append(float(linearr[2]))

# convert lists to array uT
cuT = np.zeros((n+1,n+1))
for idx in range(len(ci)):
    cuT[ci[idx],cj[idx]] = cu[idx]

# plot uT
plt.figure()
plt.pcolormesh(uT)
#plt.show()

plt.figure()
plt.pcolormesh(cuT)
plt.show()

print(np.linalg.norm(cuT-uT,'fro'))


