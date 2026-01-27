import numpy as np
from scipy import constants
import math as m

#speed of light constant
c = constants.speed_of_light 

class Cosmology:
    def __init__(self,H0,Omega_m,Omega_lambda):         #__init__ is used to initalise variables when a new instance of the class is created
        self.H0 = H0                                      #self is used so that when a new instance is created, the attributes of the class keep their original value
        self.Omega_m = Omega_m
        self.Omega_lambda = Omega_lambda
        self.Omega_k = round(1 - (self.Omega_lambda + self.Omega_m), 6)

    def computeIntegrand(self, z):
        integrand = 1 / np.sqrt(self.Omega_m * (1 + z)**3 + self.Omega_k * (1 + z)**2 + self.Omega_lambda)
        return integrand
    
    def univFlat(self):
        if self.Omega_k == 0:
            print("Universe is flat")
        else:
            print("Universe is NOT flat")
        return self.Omega_k

    def modLambda(self, new_Omega_m):
        self.Omega_m = new_Omega_m
        self.Omega_lambda = 1 - self.Omega_m
        return self.Omega_lambda
    
    def modM(self, new_Omega_lambda):
        self.Omega_lambda = new_Omega_lambda
        self.Omega_m = 1 - self.Omega_lambda
        return self.Omega_m
    
    def calcOmegamh2(self):
        h = self.H0 / 100
        Omegamh2 = self.Omega_m*h**2
        return Omegamh2
    
    def __str__(self):
        return (f"Cosmology with H0 = {self.H0}, Omega_m = {self.Omega_m}, Omega_lambda = {self.Omega_lambda}, Omega_k = {self.Omega_k}")
    
    def rectangle(self, n, z):
        """
        Computes an integral using rectangle rule.
        Approximates area under a curve as the sum of areas of rectangles space equally under a curve.

        Parameters
        ----------
        n: Whole number. Number of steps in the integration
        z: Positive Integer. Redshift 
        
        Returns
        -------
        distance: Integer. Distance calculated by rectangle rule
        """

        delta_x = z / n
        integral = 0
        for i in range(n):
            xi = i * delta_x
            integral += self.computeIntegrand(xi)

        integral = delta_x * integral
        distance = integral * ((constants.speed_of_light / 1000) / self.H0)
        #print("distance with rectangle:", distance) #in units Mpc
        
        return distance

    def trapezoid(self, n, z):
        """
        Computes an integral using the trapezoid rule.
        Approximates area under a curve as sum of areas of equally spaced trapezoids.

        Parameters
        ----------
        n: Whole number. Number of steps in the integration
        z: Positive Integer. Redshift

        Returns
        -------
        distance: Integer. Distance calculated by rectangle rule

        """

        delta_x = z / (n - 1)
        integral = 0

        for i in range(1, n - 1):
            xi = i * delta_x
            integral += self.computeIntegrand(xi)
        integral = 2 * integral

        x0 = 0
        xn_minus_one = (n - 1) * delta_x
        fx0 = self.computeIntegrand(x0)
        fxn_minus_one = self.computeIntegrand(xn_minus_one)

        integral = fx0 + integral + fxn_minus_one
        integral = (delta_x / 2) * integral

        distance = integral * ((constants.speed_of_light / 1000) / self.H0)
        #print("distance with trapezoid:",distance) #in units Mpc
        
        return distance

    def simpson(self, n, z):
        """
        Computes an integral using the Simpson rule.
        Approximates area under a curve by fitting a quadratic curve to the top of each integration region.

        Parameters
        ----------
        n: Whole number. Number of steps in the integration
        z: Positive Integer. Redshift

        Returns
        -------
        distance: Integer. Distance calculated by rectangle rule

        """

        delta_x = z / (2 * n)

        sum1 = 0
        sum2 = 0

        for i in range(n):
            xi1 = ((2 * i) + 1) * delta_x
            sum1 += self.computeIntegrand(xi1)

        for i in range(1, n):
            xi2 = (2 * i) * delta_x
            sum2 += self.computeIntegrand(xi2)

        x0 = 0
        x2n = 2 * n * delta_x
        fx0 = self.computeIntegrand(x0)
        fx2n = self.computeIntegrand(x2n)

        integral = (delta_x / 3) * (fx0 + (4 * sum1) + (2 * sum2) + fx2n)

        distance = integral * ((c / 1000) / self.H0)
        #print("distance with Simpsons:",distance) #in units Mpc

        return distance
    
    def cumulative(self, n, z):
        """
        Computes the cumulative integral of the trapezoid rule.

        Parameters
        ----------
        n: Whole number. Number of steps in the integration
        z: Positive Integer. Redshift

        Returns
        -------
        z_range: Array. An array of n equally spaced numbers from 0 to z.
        distances: Array. An array of distances calculated by the cumulative trapezoid rule.
        """

        delta_x = z / (n - 1)
        z_range = np.linspace(0, z, num = n)
        distances = np.zeros(n)

        prev_f = self.computeIntegrand(z_range[0])

        for i in range(1, n):
            curr_f = self.computeIntegrand(z_range[i])
            distances[i] = distances[i - 1] + 0.5 * delta_x * (curr_f + prev_f)
            prev_f = curr_f

        distances *= (c / 1000) / self.H0
        
        return z_range, distances
    
    def interpolate(self, array, n):
        #interpolates an array
        zmax = np.max(array)
        z_range, distances = self.cumulative(n, zmax)
        from scipy.interpolate import interp1d
        interp = interp1d(z_range, distances)
        interp_array = interp(array)
        return interp_array

    def distanceModuli(self, z_array, n):
        """
        Computes the distance modulus for different cases of omega k.
        Parameters
        ----------
        z_array: Array. Array of z values to loop through to get a range of distance moduli
        n: Whole number. Number of steps.
        Returns
        -------
        z_range: Array. An array of n equally spaced numbers from 0 to z.
        distance_mod: Array. An array of distance moduli for different z values.
        """
        distance_mod = []
        interp_array = self.interpolate(z_array, n)

        
        for i, zi in enumerate(z_array):
            x = m.sqrt(abs(self.Omega_k)) * (self.H0 * interp_array[i]) / (c/1000)
            if self.Omega_k > 0:
                S = m.sinh(x)
                Dl = (1 + zi) * ((c / 1000) / self.H0) * (1 / m.sqrt(abs(self.Omega_k))) * S
            elif self.Omega_k == 0:
                Dl = (1 + zi) * interp_array[i]
            else:
                S = m.sin(x)
                Dl = (1 + zi) * ((c / 1000) / self.H0) * (1 / m.sqrt(abs(self.Omega_k))) * S
            if Dl > 0:
                mu = (5 * np.log10(Dl)) + 25
            else:
                mu = np.nan
            distance_mod.append(mu)

        return distance_mod


if __name__ == "__main__":
    cosmo = Cosmology(70,0.3,0.7)
    z_array = np.linspace(0.01, 1.0, 50)
    z = 1
    n = 1000
    distance_mod = cosmo.distanceModuli(np.array([z]), n)
    print("Distance moduli:", distance_mod)