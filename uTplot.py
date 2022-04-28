import matplotlib.pyplot as plt
import numpy as np

with open('uT.txt') as f:
    lines = f.readlines()

ii = []
jj = []
uu = []
for line in lines:
    linearr = line.split('\t')
    ii.append(int(linearr[0]))
    jj.append(int(linearr[1]))
    uu.append(float(linearr[2]))

n = int(np.sqrt(len(ii))) - 1
uT = np.zeros((n+1,n+1))
for idx in range(len(ii)):
    uT[ii[idx],jj[idx]] = uu[idx]

plt.pcolormesh(uT)
plt.show()
