#Computer Modelling
#Aaron Liddell
#Unit 3

import matplotlib.pyplot as plt
import numpy as np
from scipy import constants
from scipy.interpolate import interp1d
import math as m

#speed of light constant
c = constants.speed_of_light 

def main():
    """
    Defines instance of Cosmology and some variables that rarely change throughout the code.
    Runs functions for testing or graph purposes.
    """

    cosmo = Cosmology(60, 0.25, 0.7)
    no_of_points = 1000
    z = 0.5
    #z_array = [0.0243, 0.385, 0.4762, 0.8745, 0.537, 0.048, 0.892, 0.9542]
    z_array = np.linspace(0, 1, 100)

    #to test the functions in the Cosmology class
    integrand = demoCosmology(cosmo, z, no_of_points) 

    #to test functions and display graphs
    graphDistanceError(cosmo, z)
    graphCumulative(cosmo, z, no_of_points)
    interpolate(cosmo, z_array, no_of_points)
    distance_mod, z_array = distanceModuli(cosmo, z_array, no_of_points)
    graphDistanceModulus(z_array, no_of_points)
    

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
    
    
def graphCumulative(cosmo, z, n):
    #graphs the cumulative distances against a range of redshifts

    z_range, distances = cosmo.cumulative(n, z)

    plt.plot(z_range, distances, label="distances against redshift values")
    plt.xlabel("Redshift")
    plt.ylabel("Distance (Mpc)")
    plt.title("Cumulative Trapezoid Distance against Redshift")
    plt.show()

def interpolate(cosmo, array, n):
    #interpolates an array

    #finds the max value in an array
    zmax = np.max(array)

    z_range, distances = cosmo.cumulative(n, zmax)
    interp = interp1d(z_range, distances)
    interp_array = interp(array)

    return interp_array

def distanceModuli(cosmo, z_array, n):
    """
        Computes the distance modulus for different cases of omega k. 

        Parameters
        ----------
        cosmo: Instance of Cosmology class.
        z_array: Array. Array of z values to loop through to get a range of distance moduli
        n: Whole number. Number of steps.

        Returns
        -------
        z_range: Array. An array of n equally spaced numbers from 0 to z.
        distance_mod: Array. An array of distance moduli for different z values.
        """
    
    distance_mod = []

    interp_array = interpolate(cosmo, z_array, n)
    
    #loops through z_array, where i is the index and zi is the value of each element of z_array
    for i, zi in enumerate(z_array):
        
        #THIS SECTION IS WRONG - unit 4 has the correct code fot this function (but i made it a method of Cosmology too)!!!!

        #calculates the argument 
        x = m.sqrt(abs(cosmo.Omega_k)) * (cosmo.H0 * interp_array[i]) / c

        if cosmo.Omega_k > 0:
            S = m.sinh(x)
            Dl = (1 + zi) * ((c / 1000) / cosmo.H0) * (1 / m.sqrt(abs(cosmo.Omega_k))) * S

        elif cosmo.Omega_k == 0:
            D_C = interp_array[i] * ((c / 1000) / cosmo.H0)
            Dl = (1 + zi) * D_C

        else:
            S = m.sin(x)
            Dl = (1 + zi) * ((c / 1000) / cosmo.H0) * (1 / m.sqrt(abs(cosmo.Omega_k))) * S
        
        #handles the case where Dl may have been 0 or negative causing a divide by 0 error
        if Dl > 0:
            mu = (5 * np.log10(Dl)) + 25
        else:
            mu = np.nan

        distance_mod.append(mu)

    print(distance_mod)
    return distance_mod, z_array

def graphDistanceModulus(z_array, n):
    #graphs distance modulus against H0, omega m and omega lambda. Each cosmological constant is varied to produces multiple plots on each graph.

    #sorts z_array from smallest to largest value. Needed to produce a smooth graph as z_array is not necessarily ordered.
    z_sorted = np.sort(z_array)


    #varying H0
    for H0_val in [68, 72, 76]:

        #new instance of Cosmology with varying H0
        cosmoH0 = Cosmology(H0_val, 0.3, 0.7)

        mu_array, z_array = distanceModuli(cosmoH0, z_sorted, n) 
        plt.plot(z_sorted, mu_array, label = f"H0 = {H0_val} ")
    
    plt.xlabel("Redshift")
    plt.ylabel("Distance Modulus")
    plt.title("Distance against Redshift for varying H0")
    plt.legend()
    plt.show()


    #varying omega m
    for m_val in [0.1, 0.3, 0.5]:

        #new instance of Comsmology with varying omega m. Ensures omega k is correctly calulated.
        cosmoM = Cosmology(72, m_val, (1 - m_val))

        mu_array, z_array = distanceModuli(cosmoM, z_sorted, n) 
        plt.plot(z_sorted, mu_array, label = f"Omega m = {m_val} ")
    
    plt.xlabel("Redshift")
    plt.ylabel("Distance Modulus")
    plt.title("Distance against Redshift for varying Omega m")
    plt.legend()
    plt.show()


    #varying omega lambda
    for l_val in [0.5, 0.7, 0.9]:

        #new instance of Cosmology with varying omega lambda. Ensures omega k is correctly calculated
        cosmoL = Cosmology(72, (1 - l_val), l_val)

         
        mu_array, z_array = distanceModuli(cosmoL, z_sorted, n)
        plt.plot(z_sorted, mu_array, label = f"Omega lambda = {l_val} ")

    plt.xlabel("Redshift")
    plt.ylabel("Distance Modulus")
    plt.title("Distance against Redshift for varying Omega lambda")
    plt.legend()
    plt.show()


def demoCosmology(cosmo, z, no_of_points):

    cosmo.rectangle(no_of_points, z)
    cosmo.trapezoid(no_of_points,z)
    cosmo.simpson(no_of_points, z)

    cosmo.cumulative(no_of_points, z)


def graphDistanceError(cosmo, z):
    """
        Graphs the absolute fractional error in the distance against the number of steps of integration, for each integration method.
        Plots log-log graphs to clearly reflect variations in the data and the number of steps.

        Parameters
        ----------
        cosmo: Instance of Cosmology class.
        z: Positive integer: Redshift value.
        """ 

    #exponentially increasing step values to plot on x axis and to find an error for each step 
    step_values = [10, 50, 100, 500, 1000, 5000, 10000]

    #get a reference "true" value
    true_distance = cosmo.simpson(100000, z)

    #create empty error and evaluation arrays to append into
    rect_errors = []
    rect_evals = []
    trap_errors = []
    trap_evals = []
    simp_errors = []
    simp_evals = []

    #calculate absolute fractional error for each integration method

    for n in step_values:
        dist = cosmo.rectangle(n, z)
        rect_errors.append(abs(dist - true_distance) / abs(true_distance))
        rect_evals.append(n)

    for n in step_values:
        dist = cosmo.trapezoid(n, z)
        trap_errors.append(abs(dist - true_distance) / abs(true_distance))
        trap_evals.append(n + 1)

    for n in step_values:
        dist = cosmo.simpson(n, z)
        simp_errors.append(abs(dist - true_distance) / abs(true_distance))
        simp_evals.append(2 * (n + 1))

    target_accuracy = 0.001


    #plotting
    plt.loglog(rect_evals, rect_errors, "o-", label="Rectangle")
    plt.loglog(trap_evals, trap_errors, "o-", label="Trapezoid")
    plt.loglog(simp_evals, simp_errors, "o-", label="Simpson")

    #plots a dashed horizontal red line at y = target accuracy 
    plt.axhline(y = target_accuracy, color = "r", linestyle = "--", linewidth = 2, label = "Target error")

    plt.xlabel("Number of evaluations")
    plt.ylabel("Absolute fractional error")
    plt.title("Absolute distance error against number of evaluations for each integration method")
    plt.legend()
    plt.show()

def graphVaryZ():
    cosmo = Cosmology(70, 0.3, 0.7)
    xVal = np.linspace(0,1,100)
    yVal = cosmo.computeIntegrand(xVal)
    plt.plot(xVal,yVal)
    plt.grid()
    plt.xlabel("z values")
    plt.ylabel("integrand")
    plt.title("integrand against z")
    plt.show()

def graphVaryOmegaM():
    xVal = np.linspace(0, 1, 100)
    Omega_m_values = [0.2, 0.3, 0.4]
    for Omega_m in Omega_m_values:
        cosmo = Cosmology(70, Omega_m, 1-Omega_m)
        yVal = cosmo.computeIntegrand(xVal)
        plt.plot(xVal, yVal, label = f"Omega m = {Omega_m}")
    plt.xlabel("z values")
    plt.ylabel("integrand")
    plt.title("Integrand for different omega m")
    plt.legend()
    plt.show()
    #changing Omega m by a small amount has a large affect on the integrand. 
    #smaller omega m results in a larger integrand for high z, but integand will converge for all omega m at low z

def graphSetOmegaM():
    cosmo = Cosmology(70, 0.3, 0.7)
    xVal = np.linspace(0, 1, 100)
    Omega_m_values = [0.2, 0.3, 0.4]

    for Omega_m in Omega_m_values:
        cosmo.modLambda(Omega_m)        #updates Omega_m in Cosmology class for cosmo instance 
        yVal = cosmo.computeIntegrand(xVal)
        plt.plot(xVal,yVal, label = f"Omega m = {Omega_m}. Omega lambda = {cosmo.Omega_lambda}")
    
    plt.xlabel("z values")
    plt.ylabel("integrand")
    plt.title("Integrand for different omega m, with one Cosmology object")
    plt.legend()
    plt.show()
    
    #appears to be no difference in the plot where multiple objects of Cosmology were made, and where only one object was made.
    #input values are the same between the two plots, so this makes sense

def printCosmo():
    c1 = Cosmology(70, 0.2, 0.8)    #assuming 0 curvature
    c2 = Cosmology(80, 0.5, 0.3)
    c3 = Cosmology(90, 0.4,0.5)
    print(c1)
    print(c2)
    print(c3)

    #Each instance of Cosmology is printed with its associated parameters that are defined in the initalising method.
    #Omega_k is calculated using Omega_lambda and Omega_m.

if __name__ == "__main__":
    
    main()




   