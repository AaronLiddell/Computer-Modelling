#Unit 4

import cosmology
import numpy as np

class Likelihood:
    #def __init__(self, redshift, mu_obs, mu_err, abs_M):
     #   self.redshift = redshift
      #  self.mu_obs = mu_obs
       # self.mu_err = mu_err
        #self.abs_M = -19.3

    def getData():
        data = np.genfromtxt("pantheon_data.txt")
        redshift = data[:,0]
        mu_obs = data[:,1]
        mu_err = data[:,2]

    def magnitude():
        distance_mod = cosmology.distanceModuli
        print(distance_mod)
        #mag = 

    #def __call__(self):

Likelihood.getData()
Likelihood.magnitude()