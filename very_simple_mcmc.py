#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 19:05:14 2019

@author: claude
"""

# Importing modules and generated data
import numpy as np
import healpy as hp

import emcee
from matplotlib import pyplot as plt

from exocartographer import IlluminationMapPosterior
from exocartographer.util import logit

import time

sim_map = np.loadtxt('sim_map_very_simple.csv', delimiter=',') # data
sim_nside = 1

hp.mollview(sim_map, min=0, max=1,
            title='Simulated Albedo Map', cmap='gist_gray') #albedo map

#%% Observations
# Set orbital properties
p_rotation = 23.934
p_orbit = 365.256363 * p_rotation
phi_orb = np.pi
inclination = np.pi/2
obliquity = 90. * np.pi/180.0
phi_rot = np.pi

# Observation schedule
day = 23.934
cadence = 1.
nobs_per_epoch = 35   #make higher to minimize uncertainty
epoch_duration = nobs_per_epoch * cadence

#epoch_starts = [30*day, 60*day, 150*day, 210*day, 250*day] for 24h observations
epoch_starts = [50*day, 120*day, 210*day, 280*day] #for 35h observations

times = np.array([])
for epoch_start in epoch_starts:
    epoch_times = np.linspace(epoch_start, epoch_start + epoch_duration,
                              nobs_per_epoch)
    times = np.concatenate([times, epoch_times])
    
measurement_std = 0.001 #to increase eventually

#%% Posterior parameters
truth = IlluminationMapPosterior(times, np.zeros_like(times),
                                 measurement_std, nside=sim_nside)


true_params = {
    'log_orbital_period':np.log(p_orbit),   
    'logit_cos_inc':logit(np.cos(inclination)),
    'logit_cos_obl':logit(np.cos(obliquity)),
    'logit_phi_orb':logit(phi_orb, low=0, high=2*np.pi),
    'logit_obl_orientation':logit(phi_rot, low=0, high=2*np.pi),
    'mu':0.5,
    'log_sigma':np.log(0.25),
    'logit_wn_rel_amp':logit(0.02),
    'logit_spatial_scale':logit(30. * np.pi/180),
    'log_error_scale': np.log(1.)}
truth.fix_params(true_params)
p = np.concatenate([[np.log(day)], sim_map])

true_lightcurve = truth.lightcurve(p)
obs_lightcurve = true_lightcurve.copy()
obs_lightcurve += truth.sigma_reflectance * np.random.randn(len(true_lightcurve))
np.savetxt('obs_lightcurve.csv', obs_lightcurve, delimiter=',')

nside = 1
logpost = IlluminationMapPosterior(times, obs_lightcurve,
                                   measurement_std,
                                   nside=sim_nside) #change to 8 for final

fix = true_params.copy()
logpost.fix_params(fix)

#%% Guess and fit
# Guess rotation period
p0 = np.random.randn(logpost.nparams)
pnames = logpost.dtype.names
p0[pnames.index('log_rotation_period')] = np.log(day)

# Defining a function that we will vary
def p_function(p_x, map_x):
    return np.concatenate([p_x, map_x])

# Checking the posterior is finite
print("log(posterior): {}".format(logpost(p_function(p0, sim_map))))

def logprior(p_x):
    log_period = p_x
    if 0 < log_period < np.log(48): #smallest observation step to double obs time
        return 0.0
    return -np.inf
def logp(p_x, map_x):
    lp = logprior(p_x)
    if not np.isfinite(lp):
        return -np.inf
    return logpost(p_function(p_x, map_x)) + lp

#%% Plotting the log prior, log posterior and log probability
#x = []
#for i in np.linspace(0, np.log(48), 1000):
    #x.append(i)
#def logprob(list_x, map_x):
    #y = []
    #for j in list_x:
        #y.append(logp(j, map_x))
    #return y


#%% Plotting the log probability with a uniform prior
#plt.figure()
#plt.plot(x, logprob(x, sim_map))
#plt.show()

#%% Gaussian prior
#def logprior_non_uni(p_x):
    #return -(np.exp(p_x)-day)**2/(2*2**2) #assume std period is 2


#def logp_non_uni(p_x, map_x):
    #lp = logprior_non_uni(p_x)
    #return logpost(p_function([p_x], map_x)) + lp

#plt.figure()
#plt.plot(x, logprior_non_uni(x))
#plt.show()

#%%
#%% MCMC
ndim = 1      #number of parameters in model
nwalkers = 100   #number of walkers is double the number of parameters, works at 50
nsteps = 10000        #number of steps to burn in, make longer for final, works at 200

program_start_time=time.time()    
sampler = emcee.EnsembleSampler(nwalkers, ndim, logp, args=[sim_map])
pos = [p0 + 1e-2*np.random.randn(ndim) for i in range(nwalkers)] #make wider for final, works at -2
mcmc = sampler.run_mcmc(pos, nsteps)
program_end_time = time.time()
print('The program ran in', program_end_time-program_start_time, 's')
# This is a test for the burn in

np.savetxt('posterior_35_very_simple_3', sampler.flatchain)

#%% Plotting the evolution 
plt.figure()
for i in range(nwalkers):
    plt.plot(sampler.chain[i,:], color='k')
    plt.xlabel('Steps')
    plt.ylabel('Log of the rotation period')
    plt.title('MCMC with flat prior')
plt.show()
pos, prob, state = mcmc

#%% Grid search for chi-square
def chi2(data, model, nparams, std=None):
    chisq=np.sum( ((data-model)/std)**2 )            
    nu=data.size-1-nparams  
    return chisq/nu

def chisq(p_x):
    return chi2(logpost.reflectance, logpost.lightcurve(p_function(p_x, sim_map)),
                1, logpost.sigma_reflectance)

#%% Broad grid search
p_search = np.arange(1,48.01, 0.01)
chisq_search = np.array([chisq([np.log(i)]) for i in p_search])
np.savetxt('chisq_search_35.csv', chisq_search, delimiter=',')
np.savetxt('p_search_35.csv', p_search, delimiter=',')

#%% Precise grid search between 23 and 25
p_search_precise = np.arange(23.7, 26.701, 0.001)
chisq_search_precise = np.array([chisq([np.log(i)]) for i in p_search_precise])
np.savetxt('chisq_search_precise_35_2.csv', chisq_search_precise, delimiter=',')
np.savetxt('p_search_precise_35_2.csv', p_search_precise, delimiter=',')