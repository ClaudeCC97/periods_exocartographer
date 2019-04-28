#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 14:37:00 2019

@author: Claude
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 07:35:15 2019

@author: Claude
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.ticker import FormatStrFormatter

#%% Data and values

# Importing the posterior
posterior = np.load('posterior_35_parallel_longer.npy')
pperiods = np.exp(posterior)

# Data for histograms
periods = pperiods.reshape(*pperiods.shape[:1], -2)

# Data for chains
chain = pperiods.reshape(*pperiods.shape[:2], -1)

# Uncertainties from MCMC for each temperature
def stats(array):
    sigma_low = np.percentile(array, 15.85)
    sigma_high = np.percentile(array, 84.15)
    mean = np.percentile(array, 50)
    return (sigma_low, mean, sigma_high)

for j in range(10):
    print(j, stats(periods[j,:]))
    
# Chi square grid search
periods_search = np.loadtxt('p_search_complete.csv')
chisq_search = np.loadtxt('chisq_search_complete.csv')    

# Obtain position of minimum chi2 for grid search 
x_min = np.argmin(chisq_search)
best_fit = periods_search[x_min]
print('Minimum chi2 =', min(chisq_search), 'located at P=', best_fit)

#%% Histogram with all temperatures

#Normalization factor
def norm_fact(i):
    return np.ones_like(periods[i,:])/len(periods[i,:])

# Parameters to differentiate temperatures
nbins = (1, 50, 200, 500, 500, 500, 500, 500, 500, 500)
#nbins = 250    
colors = ('maroon', 'coral', 'orange', 'lawngreen', 'mediumseagreen', 
          'darkslategray', 'dodgerblue', 'b', 'slateblue', 'fuchsia')
ylabels = (r'$T_1$', r'$T_2$', r'$T_3$', r'$T_4$', r'$T_5$', r'$T_6$', r'$T_7$', r'$T_8$', r'$T_9$', r'$T_{10}$')
yticks = ([0.00, 1.00], [0.00, 1.00], [0.00, 1.00],[0.00,0.02], [0.00,0.02], [0.00,0.02], [0.00,0.02], [0.00,0.02], [0.00,0.02], [0.00,0.02])
#y_limits = [1,1,1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
# Eventually set ylabels and ticks in colors, on alternating sides

# Creat figure
fig1, ax = plt.subplots(11, 1, sharex=True, figsize=(10,20))
# Remove horizontal space between axes
fig1.subplots_adjust(hspace=0)

# Histogram for each temperature
ax[0].axvline(x=23.934, color='maroon', linestyle='-', linewidth=0.5)

for i in range(10):
    ax[i].hist(periods[i,:], bins=nbins[i], weights=norm_fact(i), alpha=1, color=colors[i])
    if i % 2 == 0:
        ax[i].yaxis.tick_left()
        ax[i].yaxis.set_label_position("left")
        #ax[i].set_yticks([])
    if i % 2 == 1:
        ax[i].yaxis.tick_right()
        ax[i].yaxis.set_label_position("right")
    ax[i].tick_params(axis='y', colors=colors[i])
    ax[i].set_yticks(yticks[i])
    ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[i].set_ylabel(ylabels[i], color=colors[i], size=14)
    #ax[i].set_ylim([0, y_limits(i)])
    #ax[i].ylim([0,1])

# Chi2    
ax[10].plot(periods_search, chisq_search, linewidth=1, linestyle='-', color='k')
ax[10].set_yscale('log')
ax[10].set_ylabel(r'$\chi^2/N$', size=14)
#ax[10].set_yticks(size=14)
#ax[10].set_yticks([])
ax[10].set_xlim([0,50])
#ax[10].set_xticks(size=14)
ax[10].set_xlabel(r'Rotation period (h)', size=14)

# Show whole plot    
plt.show()

#%% Zoomed in histogram for considered temperature
fig3 = plt.figure(figsize=(10,20))

frame1=fig3.add_axes((.1,.3,.8,.6))
plt.axvline(x=23.934, color='k', linestyle='--', linewidth=0.5, label='True period')
plt.hist(periods[0], bins=1000, weights=norm_fact(0), alpha=1, color='maroon')
plt.axvspan(stats(periods[0])[0], stats(periods[0])[2], alpha=0.2, color='k', 
            label=r'1 $\sigma$ interval from MCMC')
plt.yticks([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70])
plt.ylabel('Normalized probability', size=14)
plt.xlim([0,48])
frame1.set_xticklabels([])

#Include chi2 every 0.001
subfig1 = plt.axes([0.62, 0.48, 0.20, 0.35])
#plt.yscale('log')
plt.hist(periods[0], bins=1000, weights=norm_fact(0), alpha=0.5, color='maroon')
plt.axvspan(stats(periods[0])[0], stats(periods[0])[2], alpha=0.2, color='k')
plt.ticklabel_format(axis='both', useOffset=False)
plt.xlim([23.9335, 23.9342])
plt.ylim([0, 0.1])
plt.xticks(np.arange(23.9335, 23.93421, 0.00035))
plt.yticks(color='maroon')
plt.ylabel('Normalized probability', color='maroon', size=10)


# Log chi2
frame2=fig3.add_axes((.1,.1,.8,.2))
plt.yscale('log')
plt.plot(periods_search, chisq_search, linewidth=0.5, linestyle='-', color='b', label=r'$\chi^2 /N$')
plt.plot([], [], ' ', label=r'$P_{MCMC} = 23.9339(1)$')
plt.plot([], [], ' ', label=r'$P_{\chi^2_{min}} = 23.9340$')
plt.ylabel(r'$\chi^2$ per datum', color='b', size=14)
plt.yticks(color='b')
plt.xlabel(r'Rotation period (h)', size=14)
plt.xlim([0,48])

fig3.legend(loc=[0.15, 0.65])

#%% Chain all
# Consider the different temperatures separately 
yticks2 = [23.934]

fig2, axs = plt.subplots(10, 1, sharex=True, figsize=[5,40])

for k in range(10):
    for l in range(50):    
        axs[k].plot(chain[k,l,:], linewidth=0.5, color=colors[k])
        #axs[k].axhline(y=23.934, linewidth=0.5, color='k')
        axs[k].tick_params(axis='y', colors=colors[k])
        axs[k].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        axs[k].set_yticks(yticks2)
        axs[k].set_ylabel(ylabels[k], color=colors[k], size=14)
        axs[k].set_ylim([0,50])
axs[9].set_xlabel('Steps (1/10)', size=14)
axs[9].set_xlim([0,1000])       

plt.show()

#%% Chain best
fig4 = plt.figure(figsize=[15,5])

for m in range(50):    
    plt.plot(chain[0,m,:], color='maroon', linewidth=0.5)
plt.axhline(y=23.934, linewidth=0.5, color='maroon', label='Path of walkers')
plt.axhline(y=23.934, linewidth=0.5, color='k', label='True period')
plt.ticklabel_format(axis='both', useOffset=False)
plt.xlim([0, 1000])
plt.ylim([23.933, 23.935])
plt.xticks(size=16)
plt.yticks(np.arange(23.933, 23.9351, 0.001), size=16)

plt.xlabel('Steps (1/10)', size=16)
plt.ylabel('Rotation period (h)', size=16)
plt.legend(loc=[0.77, 0.75], prop={'size': 16})

plt.show()