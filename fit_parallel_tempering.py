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

# Import the albedo map from sim_map.py and the map parameters; visualize the map
sim_map = np.loadtxt('sim_map.csv', delimiter=',')                      # simulated map
sim_nside = 1                                                           # map resolution (lowest, 12 pixels)

hp.mollview(sim_map, min=0, max=1,
            title='Simulated Albedo Map', cmap='gist_gray')             # visualize the map, in shades of gray



# Observations
# Set orbital properties, same as in sim_map.py and fit_emcee.py
p_rotation = 23.934
p_orbit = 365.256363 * p_rotation
phi_orb = np.pi
inclination = np.pi/2
obliquity = 90. * np.pi/180.0
phi_rot = np.pi

# Observation schedule
day = 23.934
cadence = 1.
nobs_per_epoch = 35                                                     # epochs of observation of 35h
epoch_duration = nobs_per_epoch * cadence
epoch_starts = [50*day, 120*day, 210*day, 280*day]                      # 4 epochs, total of 140 data points

times = np.array([])
for epoch_start in epoch_starts:
    epoch_times = np.linspace(epoch_start, epoch_start + epoch_duration,
                              nobs_per_epoch)
    times = np.concatenate([times, epoch_times])
    
# Experimental error, can be increased if the effect of the uncertainty is investigated
# Accurate results were recovered for uncertainties up to 0.01
measurement_std = 0.001                                                             



# Posterior parameters
# Identical to posterior parameters in fit_emcee.py
truth = IlluminationMapPosterior(times, np.zeros_like(times), measurement_std, nside=sim_nside)

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
truth.fix_params(true_params)                                           # fix the known parameters                      
p = np.concatenate([[np.log(day)], sim_map])                            # create a function with the map and the period

# Generate a lightcurve from the parameters
true_lightcurve = truth.lightcurve(p)                                   # lightcurve from parameters, map and period
obs_lightcurve = true_lightcurve.copy()                                 # copy the lightcurve and add noise
obs_lightcurve += truth.sigma_reflectance * np.random.randn(len(true_lightcurve))
np.savetxt('obs_lightcurve.csv', obs_lightcurve, delimiter=',')         # save the observed lightcurve as a text file

logpost = IlluminationMapPosterior(times, obs_lightcurve,               # posterior from parameters and times of observation
                                   measurement_std, nside=sim_nside)
fix = true_params.copy()
logpost.fix_params(fix)



# Guess and fit
# Guess rotation period
# Assume the chi-square grid search (see fit_emcee.py) was performed previously; use best fit as initial guess
p0 = np.random.randn(logpost.nparams)                                   # make sure each parameter has a value
pnames = logpost.dtype.names                                            # get free parameter names
p0[pnames.index('log_rotation_period')] = np.log(day)                   # initialize at chi-square minimum for period

# Define a function of the period and the map that we can easily vary; call it from the period value
def p_function(p_x):
    return np.concatenate([p_x, sim_map])

# Check the posterior is finite before initializing the optimization routine
print("log(posterior): {}".format(logpost(p_function(p0))))

# Define a uniform prior between 1h and 48h; same as in fit_emcee.py
def logprior(p_x):
    log_period = p_x
    if 0 < log_period < np.log(48): 
        return 0.0
    return -np.inf
def logp(p_x):
    lp = logprior(p_x)
    if not np.isfinite(lp):
        return -np.inf
    return logpost(p_function(p_x)) + lp



# Parallel tempering MCMC
# We still use emcee, but the parallel tempering implementation
# Note that parallel tempering is much quicker than emcee; although in my report I used the same number of steps
# for both methods, it is not required. For PT, 1000 steps would be more than enough, and 100 would probably be enough.
ndim = 1                            # number of parameters that we vary in the model (here, only the period)
nwalkers = 50                       # number of walkers (reduce from emcee to make it quicker)
nsteps = 100                        # number of steps; burn in then consider a small number of steps 
ntemps = 10                         # number of temperatures, same as in the emcee PT tutorial

program_start_time=time.time()                                                                  # start timer 
sampler = emcee.PTSampler(ntemps, nwalkers, ndim, logp, logprior)                               # sampler
pos = p0 + 1e-2*np.random.uniform(low=-1.0, high= 1.0, size=(ntemps, nwalkers, ndim))           # initial ball
for pos, prob, state in sampler.sample(pos, iterations=nsteps):
    pass
program_end_time = time.time()
print('The program ran in', program_end_time-program_start_time, 's')                           # print run time
# If wanted, could add a thin parameter to the sampler to save only some percentage of steps
# If runnning a longer chain, thin to avoid making spyder crash due to the size of the resulting file

# Save the file; can't save it as a csv so save it as a numpy file
from tempfile import TemporaryFile
posterior_35_parallel_longer = TemporaryFile()                                                  # temporary file
np.save('posterior', sampler.chain)                                                             # save the chai
