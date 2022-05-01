# Plotting strong scaling outpu 
# Creates a pcolor plot of the solution on the square domain

# (C) Tanya Wang, Sonia Reilly, Nolan Reilly, May 2022

import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from matplotlib import ticker
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import cmocean
import pickle
from mpl_toolkits.mplot3d import Axes3D



cmap = cmocean.cm.balance


### large graphs

matplotlib.rcParams.update(
    {'font.sans-serif': 'Arial', # change to anyother font you like as long as its consistent
     'font.size': 8,
     'font.family': 'Arial',
     'mathtext.default': 'regular',
     'axes.linewidth': 0.35, 
     'axes.labelsize': 8,
     'xtick.labelsize': 7,
     'ytick.labelsize': 7,     
     'lines.linewidth': 0.35,
     'legend.frameon': False,
     'legend.fontsize': 7,
     'xtick.major.width': 0.3,
     'xtick.minor.width': 0.3,
     'ytick.major.width': 0.3,
     'ytick.minor.width': 0.3,
     'xtick.major.size': 1.5,
     'ytick.major.size': 1.5,
     'xtick.minor.size': 1,
     'ytick.minor.size': 1,
    })


# read in 'strong_scale.txt'
# contains a column of i, a column of j, and a column of uT, tab separated
with open('strong_scale.txt') as f:
    lines = f.readlines()

# convert strings to lists
ntr = []
tt = []
for line in lines:
    linearr = line.split('\t')
    ntr.append(int(linearr[0]))
    tt.append(float(linearr[1]))
    



# plot tt vs ntr
ntr = np.array(ntr)
tt = np.array(tt)

f = plt.figure(figsize=[3.6, 2.8])
ax = plt.subplot()

colors2 = ['blue','orange','red']

#ax.errorbar(xs[i,:], Ds[i,:], yerr=Derr[i,:], marker = 'o',markersize=ms,label=r'$\rho=%d$'%(rhos[i]), color = colors2[i], lw=0, elinewidth = 0.75, markerfacecolor = colors2[i],capsize = 1,
 #         ecolor = 'grey', alpha = 1,mew=mkedgewidth,mec=mkedgecolor)
  
ms = 8
ax.plot(ntr, tt, 'k.--', linewidth=1,markerfacecolor = colors2[1],alpha = 1,markersize=ms,mec=colors2[1]) #, label='no flow (analytic)'
tmp=np.arange(8)+1
#ax.plot(tmp, -3*tmp+35, 'b--', linewidth=0.5) #, label='flow (analytic)'
    
#ax.legend(loc='upper left',handlelength=1, fontsize = 6,markerscale = 0.8,labelspacing = 0.5)
    

ax.set_xlabel('number of threads')
ax.set_ylabel('time (s)',labelpad = 1.5)
    #ax.set_ylabel(r'$\frac{v_e}{v_{wall}}$',labelpad = 1.5)
    
    #ax.axvline(60, dashes=(5, 2), color='gray')
    #ax.rc('xtick', labelsize=10) 
    #ax.rc('ytick', labelsize=10)
ax.grid(b=True, which='major', axis='y', color='gray', lw = 0.2, linestyle='--')
#ax.set_ylim(0,0.05)

    #ax.xaxis.set_major_locator(MultipleLocator(2))
    #ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    #ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    
ax.plot(8,tt[8-1],'o',markersize=9,markerfacecolor = 'r',mec='r')
  
    
plt.tight_layout()
#ax.set_xlim(0.1,40)
plt.title('Strong scaling of OMP')

#ms=4
#ax_new = f.add_axes([0.6, 0.6, 0.2, 0.2]) # the position of zoom-out plot compare to the ratio of zoom-in plot 
#ax_new.plot(ntr[:8], tt[:8], 'k.--', linewidth=0.5,markerfacecolor = colors2[1],alpha = 1,markersize=ms,mec=colors2[1]) 
    
plt.savefig('strong_sc.pdf', format='pdf',bbox_inches='tight')

