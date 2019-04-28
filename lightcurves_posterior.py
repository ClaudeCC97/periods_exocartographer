#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 16:38:46 2019

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

#%%
#posterior_raw = np.loadtxt('posterior_35_parallel_longer', delimiter=',').reshape(100,2000) # data
posterior_raw = np.load('posterior_35_parallel_longer.npy')
#posterior_wo_bu = posterior_raw[:,200:]
#posterior = posterior_wo_bu.flatten()
posterior = posterior_raw[0].flatten()
np.savetxt('posterior_35_very_simple_3_final', posterior)
draws = np.random.choice(posterior, size=100, replace=False, p=None)

sim_map = np.loadtxt('sim_map_very_simple.csv', delimiter=',') # data
sim_nside = 1

obs_lightcurve = np.loadtxt('obs_lightcurve.csv')
obs_lightcurve_1 = obs_lightcurve[:35]
obs_lightcurve_2 = obs_lightcurve[35:70]
obs_lightcurve_3 = obs_lightcurve[70:105]
obs_lightcurve_4 = obs_lightcurve[105:140]

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
nobs_per_epoch = 35   
epoch_duration = nobs_per_epoch * cadence

epoch_1 = np.linspace(50*day, 50*day + epoch_duration, nobs_per_epoch)
epoch_2 = np.linspace(120*day, 120*day + epoch_duration, nobs_per_epoch)
epoch_3 = np.linspace(210*day, 210*day + epoch_duration, nobs_per_epoch)
epoch_4 = np.linspace(280*day, 280*day + epoch_duration, nobs_per_epoch)

measurement_std = 0.001 #to increase eventually

truth_1 = IlluminationMapPosterior(epoch_1, np.zeros_like(epoch_1), 
                                   measurement_std, nside=sim_nside)
truth_2 = IlluminationMapPosterior(epoch_2, np.zeros_like(epoch_2), 
                                   measurement_std, nside=sim_nside)
truth_3 = IlluminationMapPosterior(epoch_3, np.zeros_like(epoch_3), 
                                   measurement_std, nside=sim_nside)
truth_4 = IlluminationMapPosterior(epoch_4, np.zeros_like(epoch_4), 
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
truth_1.fix_params(true_params)
truth_2.fix_params(true_params)
truth_3.fix_params(true_params)
truth_4.fix_params(true_params)

p = np.concatenate([[np.log(day)], sim_map])
true_lightcurve_1 = truth_1.lightcurve(p)
true_lightcurve_2 = truth_2.lightcurve(p)
true_lightcurve_3 = truth_3.lightcurve(p)
true_lightcurve_4 = truth_4.lightcurve(p)

true_posterior_1 = IlluminationMapPosterior(epoch_1, true_lightcurve_1,
                                   measurement_std, nside=sim_nside)
true_posterior_2 = IlluminationMapPosterior(epoch_2, true_lightcurve_2,
                                   measurement_std, nside=sim_nside)
true_posterior_3 = IlluminationMapPosterior(epoch_3, true_lightcurve_3,
                                   measurement_std, nside=sim_nside)
true_posterior_4 = IlluminationMapPosterior(epoch_4, true_lightcurve_4,
                                   measurement_std, nside=sim_nside)

def posterior_draw_1(draw):
    draw_lightcurve = truth_1.lightcurve(np.concatenate([[draw], sim_map]))
    return IlluminationMapPosterior(epoch_1, draw_lightcurve, measurement_std,
                                   nside=sim_nside)
def posterior_draw_2(draw):
    draw_lightcurve = truth_2.lightcurve(np.concatenate([[draw], sim_map]))
    return IlluminationMapPosterior(epoch_2, draw_lightcurve, measurement_std,
                                   nside=sim_nside)
def posterior_draw_3(draw):
    draw_lightcurve = truth_3.lightcurve(np.concatenate([[draw], sim_map]))
    return IlluminationMapPosterior(epoch_3, draw_lightcurve, measurement_std,
                                   nside=sim_nside)
def posterior_draw_4(draw):
    draw_lightcurve = truth_4.lightcurve(np.concatenate([[draw], sim_map]))
    return IlluminationMapPosterior(epoch_4, draw_lightcurve, measurement_std,
                                   nside=sim_nside)

def draw_lightcurve_1(draw):
    return truth_1.lightcurve(np.concatenate([[draw], sim_map]))
def draw_lightcurve_2(draw):
    return truth_2.lightcurve(np.concatenate([[draw], sim_map]))
def draw_lightcurve_3(draw):
    return truth_3.lightcurve(np.concatenate([[draw], sim_map]))
def draw_lightcurve_4(draw):
    return truth_4.lightcurve(np.concatenate([[draw], sim_map]))

#%% Plotting (one panel)
plt.figure()
plt.plot(true_posterior_1.times, true_lightcurve_1, color='r', linewidth=3, 
         label='True lightcurve', alpha=0.5)
plt.plot(posterior_draw_1(draws[0]).times, draw_lightcurve_1(draws[0]), color='k', 
         linewidth=0.5, label='Lightcurve drawn from posterior', alpha=0.2)
for i in draws:
    plt.plot(posterior_draw_1(i).times, draw_lightcurve_1(i), color='k', 
             linewidth=0.5, alpha=0.2)
plt.xlabel('Time (h)')
plt.ylabel('Reflectance')
plt.xlim([min(true_posterior_1.times), max(true_posterior_1.times)])
plt.legend()
plt.show()

#%% Plotting all panels together, each own scale
obs_times = np.arange(0,35.1,1)
panel = plt.figure()
ax1 = panel.add_axes([0.1, 0.75, 0.8, 0.15])
ax2 = panel.add_axes([0.1, 0.55, 0.8, 0.15])
ax3 = panel.add_axes([0.1, 0.35, 0.8, 0.15])
ax4 = panel.add_axes([0.1, 0.15, 0.8, 0.15])

ax1.set_xlim(min(true_posterior_1.times), max(true_posterior_1.times))
ax2.set_xlim(min(true_posterior_2.times), max(true_posterior_2.times))
ax3.set_xlim(min(true_posterior_3.times), max(true_posterior_3.times))
ax4.set_xlim(min(true_posterior_4.times), max(true_posterior_4.times))

ax1.plot(true_posterior_1.times, true_lightcurve_1, color='r', linewidth=3, 
         label='True lightcurve', alpha=0.5)
ax1.plot(posterior_draw_1(draws[0]).times, draw_lightcurve_1(draws[0]), color='k', 
         linewidth=0.5, label='Lightcurve drawn from posterior', alpha=0.2)
for i in draws:
    ax1.plot(posterior_draw_1(i).times, draw_lightcurve_1(i), color='k', 
             linewidth=0.5, alpha=0.2)
ax1.legend(loc='upper right')    

ax2.plot(true_posterior_2.times, true_lightcurve_2, color='r', linewidth=3, 
         alpha=0.5)
for i in draws:
    ax2.plot(posterior_draw_2(i).times, draw_lightcurve_2(i), color='k', 
             linewidth=0.5, alpha=0.2) 
    
ax3.plot(true_posterior_3.times, true_lightcurve_3, color='r', linewidth=3, 
         alpha=0.5)
for i in draws:
    ax3.plot(posterior_draw_3(i).times, draw_lightcurve_3(i), color='k', 
             linewidth=0.5, alpha=0.2) 

ax4.plot(true_posterior_4.times, true_lightcurve_4, color='r', linewidth=3, 
         alpha=0.5)
for i in draws:
    ax4.plot(posterior_draw_4(i).times, draw_lightcurve_4(i), color='k', 
             linewidth=0.5, alpha=0.2)     
panel.text(0.01, 0.5, 'Reflectance', va='center', rotation='vertical', size=14)
plt.xlabel('Time (h)', size=14)
panel.show()

#%% Plot all panels together, same scale
obs_times = np.arange(0,35.1,1)
panel = plt.figure()
ax1 = panel.add_axes([0.1, 0.75, 0.75, 0.15])
ax2 = panel.add_axes([0.1, 0.55, 0.75, 0.15])
ax3 = panel.add_axes([0.1, 0.35, 0.75, 0.15])
ax4 = panel.add_axes([0.1, 0.15, 0.75, 0.15])

# x limits to fit to data length
ax1.set_xlim(min(true_posterior_1.times), max(true_posterior_1.times))
ax2.set_xlim(min(true_posterior_2.times), max(true_posterior_2.times))
ax3.set_xlim(min(true_posterior_3.times), max(true_posterior_3.times))
ax4.set_xlim(min(true_posterior_4.times), max(true_posterior_4.times))

# Limits to have same height everywhere
ax1.set_ylim(0, 0.1)
ax2.set_ylim(0.15, 0.25)
ax3.set_ylim(0.25, 0.35)
ax4.set_ylim(0.05, 0.15)

# Ticks to the rigth
ax1.yaxis.tick_right()
ax2.yaxis.tick_right()
ax3.yaxis.tick_right()
ax4.yaxis.tick_right()

# Plots
ax1.plot(true_posterior_1.times, true_lightcurve_1, color='r', linewidth=3, 
         label='True lightcurve', alpha=0.5)
ax1.plot(posterior_draw_1(draws[0]).times, draw_lightcurve_1(draws[0]), color='k', 
         linewidth=0.5, label='Lightcurve drawn from posterior', alpha=0.2)
ax1.errorbar(true_posterior_1.times, obs_lightcurve_1, measurement_std, capthick=0,
             fmt='o', markersize=0, color='g', label='Observations')
for i in draws:
    ax1.plot(posterior_draw_1(i).times, draw_lightcurve_1(i), color='k', 
             linewidth=0.5, alpha=0.2)
ax1.legend(loc='upper right')    

ax2.plot(true_posterior_2.times, true_lightcurve_2, color='r', linewidth=3, 
         alpha=0.5)
ax2.errorbar(true_posterior_2.times, obs_lightcurve_2, measurement_std, capthick=0,
             fmt='o', markersize=0, color='g')
for i in draws:
    ax2.plot(posterior_draw_2(i).times, draw_lightcurve_2(i), color='k', 
             linewidth=0.5, alpha=0.2) 
    
ax3.plot(true_posterior_3.times, true_lightcurve_3, color='r', linewidth=3, 
         alpha=0.5)
ax3.errorbar(true_posterior_3.times, obs_lightcurve_3, measurement_std, capthick=0,
             fmt='o', markersize=0, color='g')
for i in draws:
    ax3.plot(posterior_draw_3(i).times, draw_lightcurve_3(i), color='k', 
             linewidth=0.5, alpha=0.2) 

ax4.plot(true_posterior_4.times, true_lightcurve_4, color='r', linewidth=3, 
         alpha=0.5)
ax4.errorbar(true_posterior_4.times, obs_lightcurve_4, measurement_std, capthick=0,
             fmt='o', markersize=0, color='g')
for i in draws:
    ax4.plot(posterior_draw_4(i).times, draw_lightcurve_4(i), color='k', 
             linewidth=0.5, alpha=0.2)
    
    #ax.errorbar(times[sel], logpost.reflectance[sel], 
                    #logpost.sigma_reflectance[sel],capthick=0, fmt='o', 
                   # markersize=0, color='k', label='Observations')

# Labels     
panel.text(0.95, 0.5, 'Reflectance', va='center', rotation='vertical', size=14)
panel.text(0.05, 0.825, 'Epoch 1', va='center', rotation='vertical', size=10)
panel.text(0.05, 0.625, 'Epoch 2', va='center', rotation='vertical', size=10)
panel.text(0.05, 0.425, 'Epoch 3', va='center', rotation='vertical', size=10)
panel.text(0.05, 0.225, 'Epoch 4', va='center', rotation='vertical', size=10)
plt.xlabel('Time (h)', size=14)
panel.show()