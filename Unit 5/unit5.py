#Unit 4

from cosmology import Cosmology
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import Bounds


class Likelihood:
    """
    Class for computing likelihoods and fitting cosmological models to supernova data.
    """
    def __init__(self, cosmo, abs_M = -19.3):
        """
        Initialize the Likelihood class.

        Parameters
        ----------
        cosmo : Cosmology
            An instance of the Cosmology class.
        abs_M : float, optional
            Absolute magnitude (default is -19.3).
        """
        self.cosmo = cosmo
        self.abs_M = abs_M
        self.redshift = None
        self.mu_obs = None
        self.mu_err = None

    def getData(self):
        """
        Load the supernova data from 'pantheon_data.txt' into the class attributes.

        Returns
        -------
        None
        """
        # Loads data from file. If the file is missing or formatted incorrectly, this will raise an error.
        data = np.genfromtxt("pantheon_data.txt")
        self.redshift = data[:, 0]
        self.mu_obs = data[:, 1]
        self.mu_err = data[:, 2]

    def magnitude(self, n):
        """
        Calculate the model magnitudes for the current cosmology and redshifts.

        Parameters
        ----------
        n : int
            Number of integration points for distance calculation.

        Returns
        -------
        mag : np.ndarray
            Array of model magnitudes for each redshift.
        """
        if self.redshift is None:
            # If data hasn't been loaded yet, load it now.
            self.getData()

        distance_mod = self.cosmo.distanceModuli(self.redshift, n)
        mag = np.array(distance_mod) + self.abs_M
        
        return mag
    
    def likelihood(self, mag):
        """
        Compute the log-likelihood for the given model magnitudes.

        Parameters
        ----------
        mag : np.ndarray
            Model magnitudes for each redshift.

        Returns
        -------
        logP : float
            The log-likelihood value.
        """
        # Compute the sum of squared residuals, weighted by the errors.
        sum = 0
        for i in range(len(mag)):
            sum += ((self.mu_obs[i] - mag[i])**2 / self.mu_err[i]**2)
        
        logP = -1/2 * sum

        return logP

    def convergence(self):
        """
        Plot the convergence of the log-likelihood as a function of the number of integration points.

        Returns
        -------
        None
        """
        # Test convergence for a range of integration points.
        N_range = [50, 100, 200, 500, 1000, 2000, 5000]
        
        logP_list = []
        for i in range(len(N_range)):
            mag_i = self.magnitude(N_range[i])
            logP_i = self.likelihood(mag_i)
            logP_list.append(logP_i)

        self.convergencePlot(logP_list, N_range)        

    def convergencePlot(self, logP_list, N):
        """
        Plot the log-likelihood and its change as a function of integration points.

        Parameters
        ----------
        logP_list : list or np.ndarray
            List of log-likelihood values for each N.
        N : list or np.ndarray
            List of integration point values.

        Returns
        -------
        None
        """
        logP_list = np.array(logP_list)
        dlogP_list = np.abs(np.diff(logP_list))

        plt.subplot(1, 2, 1)
        plt.plot(N, logP_list)
        plt.xlabel("N (integration points)")
        plt.ylabel("log-likelihood")
        plt.title("Log-likelihood vs N")

        plt.subplot(1, 2, 2)
        plt.plot(N[1:], dlogP_list)  #removes the first element in the array, as it has no value before it to compare to
        plt.xlabel("N (integration points)")
        plt.ylabel("Change in log-likelihood")
        plt.ylim(0, 1)
        plt.title("Change in log-likelihood vs N")

        plt.show()

    def optimise(self, n, model_type="standard"):
        """
        Find the best-fit cosmological parameters by maximizing the likelihood (minimizing negative log-likelihood).

        Parameters
        ----------
        n : int
            Number of integration points for distance calculation.
        model_type : str, optional
            Which model to use: 'standard' (default, 3 parameters) or 'no_lambda' (2 parameters, Omega_lambda=0).

        Returns
        -------
        minimized_params.x : np.ndarray
            Best-fit parameter values.
        minimized_params.fun : float
            Best-fit negative log-likelihood value.
        """
        # Use a lambda to ensure the correct model_type is passed to negLogLikelihood.
        if model_type == "standard":
            initial_guess = [70, 0.3, 0.7]  #Ho, Omega m, Omega lambda
            bounds = Bounds([67, 0.308, 0.69], [74, 0.322, 0.73])
            minimized_params = minimize(lambda params: self.negLogLikelihood(params, n, model_type = "standard"), initial_guess, bounds = bounds)

        elif model_type == "no_lambda":
            initial_guess = [70, 0.3]
            bounds = Bounds([67, 0.308], [74, 0.322])
            minimized_params = minimize(lambda params: self.negLogLikelihood(params, n, model_type = "no_lambda"), initial_guess, bounds = bounds)
        
        print(f"Best-fit parameters for {model_type}: {minimized_params.x}")
        print(f"Best-fit negative log-likelihood: {minimized_params.fun}")

        return minimized_params.x, minimized_params.fun

    def negLogLikelihood(self, params, n, model_type = "standard"):
        """
        Compute the negative log-likelihood for given parameters and model type.

        Parameters
        ----------
        params : list or np.ndarray
            Model parameters (length 3 for 'standard', length 2 for 'no_lambda').
        n : int
            Number of integration points for distance calculation.
        model_type : str, optional
            Which model to use: 'standard' or 'no_lambda'.

        Returns
        -------
        float
            Negative log-likelihood value.
        """
        if model_type == "standard":
            H0, Omega_m, Omega_lambda = params

        elif model_type == "no_lambda":
            H0, Omega_m = params
            Omega_lambda = 0
            
        cosmo = Cosmology(H0, Omega_m, Omega_lambda)
        self.cosmo = cosmo
        
        mag = self.magnitude(n)
        return -self.likelihood(mag)
    
    def plotDataBestFit(self, n):
        """
        Plot the data with error bars and the best-fit model, and plot the residuals.

        Parameters
        ----------
        n : int
            Number of integration points for distance calculation.

        Returns
        -------
        None
        """
        self.getData()
        best_params = self.optimise(n)
        H0, Omega_m, Omega_lambda = best_params

        best_cosmo = Cosmology(H0, Omega_m, Omega_lambda)
        best_likeliood = Likelihood(best_cosmo)

        model_mag = self.magnitude(n)

        #data and best fit curve
        plt.errorbar(self.redshift, self.mu_obs, yerr = self.mu_err, fmt = "o", label = "data")
        plt.plot(self.redshift, model_mag, label = "Best fit model", zorder = 10)
        plt.xlabel("Redshift")
        plt.ylabel("Distance Modulus")
        plt.legend()
        plt.title("Data with Best Fit Model")
        plt.show()

        #residuals
        residuals = []
        for i in range(len(model_mag)):
            residual_i = (self.mu_obs[i] - model_mag[i]) / self.mu_err[i]
            residuals.append(residual_i)

        plt.errorbar(self.redshift, residuals, yerr = self.mu_err, fmt = "o")
        plt.axhline(0, color = "black", linestyle = ":", linewidth = 1)
        plt.xlabel("Redshift")
        plt.ylabel("Residuals of Distance Moduli")
        plt.title("Residuals of Distance Moduli against Redshift")
        plt.show()

        print(f"mean of residuals {np.mean(residuals)}")
        print(f"standard deviation of residuals {np.std(residuals)}")

    def __call__(self, params, n, model_type = "standard"):
        """
        Callable interface for the likelihood, allowing use in optimizers.

        Parameters
        ----------
        params : list or np.ndarray
            Model parameters (length 3 for 'standard', length 2 for 'no_lambda').
        n : int
            Number of integration points for distance calculation.
        model_type : str, optional
            Which model to use: 'standard' or 'no_lambda'.

        Returns
        -------
        float
            Negative log-likelihood value.
        """
        if model_type == "standard":
            H0, Omega_m, Omega_lambda = params

        elif model_type == "no_lambda":
            H0, Omega_m = params
            Omega_lambda = 0

        cosmo = Cosmology(H0, Omega_m, Omega_lambda)
        self.cosmo = cosmo
        mag = self.magnitude(n)
        
    def likelihoodGrid3d(self, n):
        points = 10

        p0_min, p0_max = 0.2, 0.4  #omegam
        p0 = np.linspace(p0_min, p0_max, points)

        p1_min, p1_max = 0.6, 0.8  #omega lambda
        p1 = np.linspace(p1_min, p1_max, points)

        p2_min, p2_max = 71.3, 72.8  #H0
        p2 = np.linspace(p2_min, p2_max, points)

        like_grid_3d = np.zeros((points, points, points))

        for i in range(points):
            for j in range(points):
                for k in range(points):
                    params = p2[i], p0[j], p1[k] 
                    loglike = self.negLogLikelihood(params, n, "standard")
                    like_grid_3d[i, j, k] = loglike
                    
                    #if p0[j] + p1[k] <= 1.0:
                     #   params = p2[i], p0[j], p1[k] 
                      #  loglike = self.negLogLikelihood(params, n, "standard")
                       # like_grid_3d[i, j, k] = loglike
                    #else:
                     #   like_grid_3d[i, j, k] = np.nan

        return like_grid_3d

    def marginalizedLikelihoods(self, G3d):
        G3d_max = np.nanmax(G3d, axis=0)

        vals = G3d[~np.isnan(G3d)]
        print("min logL:", np.min(vals), "max logL:", np.max(vals))
        print("max - min:", np.max(vals) - np.min(vals))




        #Marginalize 2d likelihoods
        #marginalize over H0
        margin_H0 = np.nansum(np.exp(G3d - G3d_max), axis = 0)
        print(margin_H0)

        #marginalize over omega m
        margin_omega_m = np.nansum(np.exp(G3d - G3d_max), axis = 1)

        #marginalize over omega lambda
        margin_omega_lambda = np.nansum(np.exp(G3d - G3d_max), axis = 2)


        #marginalized 1d likelihoods
        #margin. over H0 and omega m
        margin_H0_omega_m = np.nansum(np.exp(G3d - G3d_max), axis = (0, 1))

        #margin. over H0 and omega lambda
        margin_H0_omega_lambda = np.nansum(np.exp(G3d-G3d_max), axis = (0, 2))

        #margin. over omega m and omega lambda
        margin_omega_m_omega_lambda = np.nansum(np.exp(G3d - G3d_max), axis = (1, 2))


        #plot the 2d results
        #plt.figure()
        plt.imshow(margin_H0)
        plt.xlabel("Omega m")
        plt.ylabel("Omega lambda")
        plt.title("Marginalized Likelihood over H0")
        plt.colorbar(label = "Likelihood")
        plt.show()

        #plt.figure()
        #plt.imshow(margin_omega_m)
        #plt.xlabel("H0")
        #plt.ylabel("Omega lambda")
        #plt.title("Marginalized Likelihood over omega m")
        #plt.colorbar(label = "Likelihood")
        #plt.show()


def main():
    H0 = 70
    Omega_m = 0.3
    Omega_lambda = 0.7
    n = 2500 #number of integration steps

    cosmo = Cosmology(H0, Omega_m, Omega_lambda)
    like = Likelihood(cosmo)

    #standard model
    #best_params_std, best_fun_std = like.optimise(n, model_type = "standard")

    #no lambda model
    #best_params_nolambda, best_params_nolambda = like.optimise(n, model_type = "no_lambda")

    like_grid_3d = like.likelihoodGrid3d(n)
    like.marginalizedLikelihoods(like_grid_3d)
   
if __name__ == "__main__":
    main()
