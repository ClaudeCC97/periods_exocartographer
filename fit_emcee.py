#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 19:05:14 2019

@author: claude
"""
# We perform a Metropolis-Hastings Markov chain Monte Carlo to fit for the rotation period
# This method is not the most efficient; it is presented here for completeness

# Importing modules
import numpy as np
import healpy as hp
import emcee
from matplotlib import pyplot as plt
from exocartographer import IlluminationMapPosterior
from exocartographer.util import logit
import time

# Import the data generated in sim_map.py, with its resolution
sim_map = np.loadtxt('sim_map.csv', delimiter=',')                      # data
sim_nside = 1                                                           # resolution (currently, lowest possible, 12 pixels)

# Visualize the albedo map
hp.mollview(sim_map, min=0, max=1,
            title='Simulated Albedo Map', cmap='gist_gray')             # albedo map in shades of gray



# Observations and observational parameters
# Set orbital properties
# Here, the orbital properties are the same as in those used by Farr et al. (2018) in their exocartographer paper
p_rotation = 23.934                                                     # rotation period in hours
p_orbit = 365.256363 * p_rotation                                       # orbital period in hours
phi_orb = np.pi                                 
inclination = np.pi/2
obliquity = 90. * np.pi/180.0
phi_rot = np.pi

# Observation schedule
day = 23.934
cadence = 1.
nobs_per_epoch = 35                                                     # number of observations per epochs of 35h
epoch_duration = nobs_per_epoch * cadence
epoch_starts = [50*day, 120*day, 210*day, 280*day]                      # 4 epochs of observation; total of 140 data points

times = np.array([])                                                    # creating a time array for observations
for epoch_start in epoch_starts:
    epoch_times = np.linspace(epoch_start, epoch_start + epoch_duration, nobs_per_epoch)
    times = np.concatenate([times, epoch_times])
    
measurement_std = 0.001                                                 # standard deviation of measurements from truth

# Posterior parameters
truth = IlluminationMapPosterior(times, np.zeros_like(times), measurement_std, nside=sim_nside)
# Parameters for the gaussian process and the maps; same as those used in sim_map.py
# We fix all parameters except for the rotation period, and assume other parameters are known
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
truth.fix_params(true_params)                                           # fixing the parameters with the measurements
p = np.concatenate([[np.log(day)], sim_map])                            # create an array with the map and the rotation period

# Generate and save a lightcurve
true_lightcurve = truth.lightcurve(p)                                   # lightcurve from known parameters, map and period
obs_lightcurve = true_lightcurve.copy()                                 # copy the lightcurve and add noise (below)
obs_lightcurve += truth.sigma_reflectance * np.random.randn(len(true_lightcurve))
np.savetxt('obs_lightcurve.csv', obs_lightcurve, delimiter=',')         # save the lightcurve with noise as a csv file

# Posterior from lightcurve
logpost = IlluminationMapPosterior(times, obs_lightcurve,               # posterior from lightcurve and parameters 
                                   measurement_std,
                                   nside=sim_nside)
fix = true_params.copy()
logpost.fix_params(fix)                                                 # fix the parameters to the true parameters



# Chi-square minimization and initial guess
# Grid search for chi-square and period
# Define a general chi-square per datum function
def chi2(data, model, nparams, std=None):
    chisq=np.sum( ((data-model)/std)**2 )            
    nu=data.size-1-nparams  
    return chisq/nu

# Define a specific chi-square function to fit for the period
def chisq(p_x):
    return chi2(logpost.reflectance, logpost.lightcurve(p_function(p_x, sim_map)),
                1, logpost.sigma_reflectance)

# Grid search
p_search = np.arange(1, 48.01, 0.001)                                   # period values for every 0.001 between 1h and 48h
chisq_search = np.array([chisq([np.log(i)]) for i in p_search])         # array of chi-squares for the above period values
np.savetxt('chisq_search.csv', chisq_search, delimiter=',')             # save the chi-squares as a csv file
np.savetxt('p_search.csv', p_search, delimiter=',')                     # save the periods as a csv file

# Guess rotation period
# We use the chi-square minimizing value of the period as an initial guess (i.e. we initialize the walkers there)
p0 = np.random.randn(logpost.nparams)                                   # make sure all parameters in logpost have a value
pnames = logpost.dtype.names
p0[pnames.index('log_rotation_period')] = np.log(day)                   # use the chi-square minimum as an initial guess

# We define a function that depends only on the map and the period that we can vary easily
def p_function(p_x, map_x):
    return np.concatenate([p_x, map_x])

# Check that the posterior is finite before initializing the optimization routine
print("log(posterior): {}".format(logpost(p_function(p0, sim_map)))
      
# Defining a prior
# We want a uniform prior between 1h (since we take data every hour) and 48h (double the period)
def logprior(p_x):
    log_period = p_x
    if 0 < log_period < np.log(48):                                     # prior uniformly 0 in log space between 1h and 48h
        return 0.0
    return -np.inf
def logp(p_x, map_x):
    lp = logprior(p_x)
    if not np.isfinite(lp):                                             # prior - inf outside of the interval [1h, 48h]
        return -np.inf
    return logpost(p_function(p_x, map_x)) + lp




# MCMC
# Although Metropolis-Hastings MCMC is not the most efficient way to recover the period, it is provided for completeness
# Parameters for emcee
ndim = 1                                                                # number of parameters in model (only the period)
nwalkers = 100                                                          # number of walkers (works at 50)
nsteps = 10000                                                          # number of steps (burn in before)

program_start_time=time.time()                                          # start the timer
sampler = emcee.EnsembleSampler(nwalkers, ndim, logp, args=[sim_map])   # sampler
pos = [p0 + 1e-2*np.random.randn(ndim) for i in range(nwalkers)]        # initial position of the walkers around chi2 min.
mcmc = sampler.run_mcmc(pos, nsteps)                                    # initialize emcee
program_end_time = time.time()      
print('The program ran in', program_end_time-program_start_time, 's')   # print run time

pos, prob, state = mcmc                                                 # if burn in, initialize mcmc from pos
np.savetxt('posterior', sampler.flatchain)                              # save the posterior; right shape to make histogram

# Plot the path of the walkers to check the evolution/as a sanity check
# This plot should not be presented in a final report before including the prior as a shaded region
plt.figure()
for i in range(nwalkers):
    plt.plot(sampler.chain[i,:], color='k')                             # path of each walker
    plt.xlabel('Steps')                                                 # x-axis label
    plt.ylabel('Log of the rotation period')                            # y-axis label
    plt.title('MCMC with uniform prior between 1h and 48h')             # label
plt.show()
