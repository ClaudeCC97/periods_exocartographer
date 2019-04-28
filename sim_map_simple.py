#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 12:40:28 2018

@author: claude
"""

#Importing modules
import numpy as np
import healpy as hp

from scipy.optimize import minimize

from matplotlib import pyplot as plt

from exocartographer.gp_map import draw_map
from exocartographer import IlluminationMapPosterior
from exocartographer.util import logit, inv_logit

import time

sim_nside = 2  # map resolution

# Gaussian process properties
sim_wn_rel_amp = 0.02
sim_length_scale = 30. * np.pi/180
sim_albedo_mean = .5
sim_albedo_std = 0.25

# Draw a simplified albedo map
while True:
    simulated_map = draw_map(1, sim_albedo_mean,
                             sim_albedo_std, sim_wn_rel_amp,
                             sim_length_scale)
    if min(simulated_map) > 0 and max(simulated_map) < 1:
        break

hp.mollview(simulated_map, min=0, max=1,
            title='Simulated Albedo Map', cmap='gist_gray')

np.savetxt('sim_map_very_simple.csv', simulated_map, delimiter=',')