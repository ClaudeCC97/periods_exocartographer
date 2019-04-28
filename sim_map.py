#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 12:40:28 2018

@author: claude
"""
# We want to simulate a very simple albedo map using healpy and exocartographer

#Importing modules
import numpy as np
import healpy as hp
from exocartographer.gp_map import draw_map

# Map resolution
# The resolution is currently set to 1 (lowest) since we fit only only for the period
# With this resolution, get 12 pixels
# For the conversion between resolution and number of pixels, see exocartographer page on GitHub
sim_nside = 1 

# Gaussian process properties with which the map is generated
# Those properties were kept the same as in the exocartographer tutorial
sim_wn_rel_amp = 0.02                   # Relative amplitude of white noise
sim_length_scale = 30. * np.pi/180      # Length scale of the albedo features, in degrees
sim_albedo_mean = .5                    # Mean albedo, set to 0.5 for convenience (lower for Earth)
sim_albedo_std = 0.25                   # Standard deviation of albedo distribution

# Draw a simplified albedo map, constrained between 0 and 1
while True:
    simulated_map = draw_map(sim_nside, sim_albedo_mean,
                             sim_albedo_std, sim_wn_rel_amp,
                             sim_length_scale)
    if min(simulated_map) > 0 and max(simulated_map) < 1:
        break

# Visualize the map; the color can be changed by changing cmap
hp.mollview(simulated_map, min=0, max=1,
            title='Simulated Albedo Map', cmap='gist_gray')

# Save the albedo map in a csv file
np.savetxt('sim_map.csv', simulated_map, delimiter=',')
