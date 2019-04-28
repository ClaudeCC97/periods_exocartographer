#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 17:03:13 2019

@author: Claude
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.ticker import FormatStrFormatter

#%% Data and values

# Importing the posterior
posterior_001 = np.load('posterior_35_parallel_test.npy')
posterior_005 = np.load('posterior_35_parallel_005.npy')
posterior_01 = np.load('posterior_35_parallel_01.npy')
posterior_05 = np.load('posterior_35_parallel_05.npy')

pperiods_001 = np.exp(posterior_001)
pperiods_005 = np.exp(posterior_005)
pperiods_01 = np.exp(posterior_01)
pperiods_05 = np.exp(posterior_05)

# Data for histograms
periods_001 = pperiods_001.reshape(*pperiods_001.shape[:1], -2)
periods_005 = pperiods_005.reshape(*pperiods_005.shape[:1], -2)
periods_01 = pperiods_01.reshape(*pperiods_01.shape[:1], -2)
periods_05 = pperiods_05.reshape(*pperiods_05.shape[:1], -2)

# Data for chains
chain_001 = pperiods_001.reshape(*pperiods_001.shape[:2], -1)
chain_005 = pperiods_005.reshape(*pperiods_005.shape[:2], -1)
chain_01 = pperiods_01.reshape(*pperiods_01.shape[:2], -1)
chain_05 = pperiods_05.reshape(*pperiods_05.shape[:2], -1)

# Uncertainties from MCMC for each temperature
def stats(array):
    sigma_low = np.percentile(array, 15.85)
    sigma_high = np.percentile(array, 84.15)
    mean = np.percentile(array, 50)
    return (sigma_low, mean, sigma_high)

print('001', stats(periods_001[0,:]))
print('005', stats(periods_005[0,:]))
print('01', stats(periods_01[0,:]))
print('05', stats(periods_05[0,:]))
    
# Chi square grid search
periods_search = np.loadtxt('p_search_complete.csv')
chisq_search = np.loadtxt('chisq_search_complete.csv')    

# Obtain position of minimum chi2 for grid search 
x_min = np.argmin(chisq_search)
best_fit = periods_search[x_min]
print('Minimum chi2 =', min(chisq_search), 'located at P=', best_fit)

#%% Histogram all

#Normalization factor
def norm_fact(array):
    return np.ones_like(array)/len(array)

# Parameters to differentiate temperatures
nbins = (50, 100, 200, 500)
yticks = ([0.00, 1], [0.00, 1], [0.00, 1], [0.00, 0.2])
ylabels = (r'$\sigma R = 0.001$', r'$\sigma R = 0.005$', r'$\sigma R = 0.01$', r'$\sigma R = 0.05$')

# Array with 4 different runs
periods = (periods_001[0], periods_005[0], periods_01[0], periods_05[0])
colors = ('coral', 'mediumseagreen', 'dodgerblue', 'fuchsia')

# Creat figure
fig1, ax = plt.subplots(5, 1, sharex=True, figsize=(20,20))
# Remove horizontal space between axes
fig1.subplots_adjust(hspace=0)

# Histogram for each temperature
for i in range(4):
    ax[i].hist(periods[i], bins=nbins[i], weights=norm_fact(periods[i]), alpha=0.5, color=colors[i])
    #ax[i].set_yticks([])
    ax[i].set_ylim(yticks[i])
    if i % 2 == 0:
        ax[i].yaxis.tick_left()
        ax[i].yaxis.set_label_position("left")
        #ax[i].set_yticks([])
    if i % 2 == 1:
        ax[i].yaxis.tick_right()
        ax[i].yaxis.set_label_position("right")
    ax[i].tick_params(axis='y', colors=colors[i])
    ax[i].set_yticks(yticks[i])
    ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax[i].set_ylabel(ylabels[i], color=colors[i], size=14)    
    #ax[i].ylim([0,1])

# Chi2    
ax[4].plot(periods_search, chisq_search, linewidth=0.5, linestyle='-', color='k')
ax[4].set_yscale('log')
ax[4].set_ylabel(r'$\chi^2/N$', size=14)
ax[4].set_xlim([0,50])
ax[4].set_xlabel(r'Rotation period (h)', size=14)

# Show whole plot    
plt.show()

#%% Histogram - zoomed in on three converged chains
# Creat figure
ytickss = [0, 0.2, 0.4, 0.6, 0.8]
nbins = [250, 175, 100]
#periods_001 = pperiods_001.reshape(*pperiods_001.shape[:1], -2)
#periods_005 = pperiods_005.reshape(*pperiods_005.shape[:1], -2)
#periods_01 = pperiods_01.reshape(*pperiods_01.shape[:1], -2)
#periods_05 = pperiods_05.reshape(*pperiods_05.shape[:1], -2)

periodss_001 = periods_001[0][(periods_001[0]>=23.93)*(periods_001[0]<=23.938)]
periodss_005 = periods_005[0][(periods_005[0]>=23.93)*(periods_005[0]<=23.938)]
periodss_01 = periods_01[0][(periods_01[0]>=23.93)*(periods_01[0]<=23.938)]

periodss = [periodss_001, periodss_005, periodss_01]

fig2, axs = plt.subplots(3, 1, sharex=True, figsize=(10,15))
# Remove horizontal space between axes
fig2.subplots_adjust(hspace=0)

# Histogram for each temperature

axs[0].hist(periodss[0], bins=nbins[0], weights=norm_fact(periodss[0]), alpha=0.8, color=colors[0])
axs[0].axvspan(stats(periodss[0])[0], stats(periodss[0])[2], alpha=0.2, color=colors[0], 
   label=r'1 $\sigma$ for $\sigma$ R = 0.001')
#ax[i].set_yticks([])
axs[0].tick_params(axis='y', colors=colors[0])
#axs[j].set_yticks(ytickss)
axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
axs[0].set_ylabel(ylabels[0], color=colors[0], size=14)

axs[1].hist(periodss[1], bins=nbins[1], weights=norm_fact(periodss[1]), alpha=0.8, color=colors[1])
axs[1].axvspan(stats(periodss[1])[0], stats(periodss[1])[2], alpha=0.2, color=colors[1], 
   label=r'1 $\sigma$ for $\sigma$ R = 0.005')
#ax[i].set_yticks([])
axs[1].tick_params(axis='y', colors=colors[1])
#axs[j].set_yticks(ytickss)
axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
axs[1].set_ylabel(ylabels[1], color=colors[1], size=14) 
    
axs[2].hist(periodss[2], bins=nbins[2], weights=norm_fact(periodss[2]), alpha=0.8, color=colors[2])
axs[2].axvspan(stats(periodss[2])[0], stats(periodss[2])[2], alpha=0.2, color=colors[2], 
   label=r'1 $\sigma$ for $\sigma$ R = 0.01')
#ax[i].set_yticks([])
axs[2].tick_params(axis='y', colors=colors[2])
#axs[j].set_yticks(ytickss)
axs[2].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
axs[2].set_ylabel(ylabels[2], color=colors[2], size=14) 

axs[2].set_xlim([23.93,23.938])
axs[2].set_xlabel(r'Rotation period (h)', size=14)

# Show whole plot 
fig2.legend(loc=[0.6, 0.7])   
plt.show()

#%% Plot of evolution of uncertainties
R = [0.001, 0.005, 0.01]
unc = [stats(periodss[0])[2]-stats(periodss[0])[0], stats(periodss[1])[2]-stats(periodss[1])[0], 
       stats(periodss[2])[2]-stats(periodss[2])[0]]

plt.figure(figsize=(10,10))
plt.scatter(R[0], unc[0], marker='o', color=colors[0])
plt.scatter(R[1], unc[1], marker='o', color=colors[1])
plt.scatter(R[2], unc[2], marker='o', color=colors[2])
plt.xlim(0, 0.011)
plt.xlabel(r'$\sigma$ R', size=16)
plt.xticks(size=16)
plt.ylim(0, 0.0025)
plt.ylabel(r'$\sigma$ P', size=16)
plt.yticks(size=16)
plt.grid()
plt.show()

#%% Chain
chains = [chain_001[0], chain_005[0], chain_01[0], chain_05[0]]

# Consider the different temperatures separately 
fig2, axs = plt.subplots(4, 1, sharex=True, figsize=[5,40])

for k in range(4):
    for l in range(50):    
        axs[k].plot(chains[k][l,:], linewidth=0.5, color=colors[k])
        axs[k].axhline(y=23.934, linewidth=0.5, color='k')
        axs[k].set_yticks([])
        axs[k].set_ylim([0,48])
axs[3].set_xlabel('Steps', size=16)
axs[3].set_xlim([0,100])       

plt.show()