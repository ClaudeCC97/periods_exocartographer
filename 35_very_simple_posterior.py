#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 07:35:15 2019

@author: Claude
"""

import numpy as np
from matplotlib import pyplot as plt

#%% Data and values

posterior = np.loadtxt('posterior_35_very_simple')
#posterior_without_burn_in = posterior[2500:]
periods = np.exp(posterior)
normalization_factor = np.ones_like(periods)/len(periods)
nbins = 250

sigma_low = np.percentile(periods, 15.85)
sigma_high = np.percentile(periods, 84.15)
sigma_low_low = np.percentile(periods, 2.25)
sigma_high_high = np.percentile(periods, 97.75)
mean = np.percentile(periods, 50)
print(sigma_low, mean, sigma_high)

periods_search = np.loadtxt('p_search_35.csv')
chisq_search = np.loadtxt('chisq_search_35.csv')
periods_search_precise = np.loadtxt('p_search_precise_35.csv')
chisq_search_precise = np.loadtxt('chisq_search_precise_35.csv')

#%% Obtain position of minimum chi2 for grid search 

# Broad search - every 0.01
x_min_broad = np.argmin(chisq_search)
best_fit_broad = periods_search[x_min_broad] 

# Precise search - every 0.001 between 23 and 25
x_min = np.argmin(chisq_search_precise)
best_fit = periods_search_precise[x_min]
print('Minimum chi2 =', min(chisq_search_precise), 'located at P=', best_fit)

#%% Figure with chi2 calculated every 0.01 h - Wide
fig = plt.figure(figsize=(10,20))

frame1=fig.add_axes((.1,.4,.8,.5))
plt.axvline(x=23.934, color='r', linestyle='--', linewidth=0.5, label='True period')
plt.hist(periods, bins=nbins, weights=normalization_factor, alpha=1, color='k')
plt.axvspan(sigma_low, sigma_high, alpha=0.2, color='k', label=r'1 $\sigma$ interval')
#plt.axvspan(sigma_low_low, sigma_high_high, alpha=0.1, color='b', label=r'2 $\sigma$')
plt.yticks([0.10, 0.20, 0.30, 0.40])
plt.ylabel('Normalized probability', size=14)
plt.xlim([0,48])
plt.plot([], [], ' ', label=r'P = $23.9(6)$')
frame1.set_xticklabels([])
#plt.title('Rotation periods obtained from MCMC')

#Include chi2 every 0.001
subfig1 = plt.axes([0.6, 0.48, 0.20, 0.35])
#plt.yscale('log')
plt.hist(periods, bins=nbins, weights=normalization_factor, alpha=1, color='k')
#plt.plot(periods_search_precise, chisq_search_precise, linewidth=1, color='b', linestyle='-', label=r'$\chi^2$')
#plt.axhspan(min(chisq_search_precise), min(chisq_search_precise)+1, alpha=0.2, color='k')
plt.xlim([23.4, 24.6])
plt.ylim([0, 1])
plt.xticks(np.arange(23.5, 24.6, 0.5))
plt.ylabel('Normalized probability', size=10)

subfig2 = subfig1.twinx()
plt.yscale('log')
#plt.hist(periods, bins=nbins, weights=normalization_factor, alpha=1, color='k')
plt.plot(periods_search_precise, chisq_search_precise, linewidth=1, color='b', linestyle='-', label=r'$\chi^2$')
plt.axhspan(min(chisq_search_precise), min(chisq_search_precise)+1, alpha=0.2, color='k')
plt.ylim([0.0001, 1000.1])
plt.ylabel(r'$\chi^2$ per datum', size=10, color='b')
plt.yticks(color='b')
#plt.xlim(0, 0.2)

# Broad chi2 - every 0.01
#frame2=fig.add_axes((.1,.25,.8,.15))
#plt.plot(periods_search, chisq_search, linewidth=0.5, linestyle='--', color='b', label=r'$\chi^2$')
#plt.ylabel(r'$\chi^2$')
#plt.yticks(np.arange(0, 999, 500))
#plt.xlim([0,48])

# Log chi2
frame2=fig.add_axes((.1,.1,.8,.3))
plt.yscale('log')
plt.plot(periods_search, chisq_search, linewidth=0.5, linestyle='-', color='b')
#plt.axhspan(min(chisq_search_precise), min(chisq_search_precise)+1, alpha=0.2, color='k')
plt.ylabel(r'$\chi^2$ per datum', size=14)
plt.xlabel(r'Rotation period (h)', size=14)
plt.xlim([0,48])

fig.legend(loc=[0.15, 0.70])

plt.savefig('35_posterior_chi2.pdf')