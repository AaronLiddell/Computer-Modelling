#Unit 6

from cosmology import Cosmology
from likelihood import Likelihood
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import Bounds




class Metropolis:
    """
    Run a Metropolis MCMC analysis for cosmological parameters.

    The sampler uses a Gaussian random-walk proposal in the parameter order
    [H0, Omega_m, Omega_lambda], stores the chain and log-likelihood trace,
    and provides simple plotting/analysis utilities.
    """

    #def __init__(self, like, loglike_fn, initial_params, proposal_scales, n_steps, rng=None):
     #   self.like = like
      #  self.loglike_fn = loglike_fn
       # self.current_params = np.array(initial_params)
        #self.proposal_scales = np.array(proposal_scales)  #0.3, 0.07, 0.1
        #self.n_steps = n_steps
        #self.rng = np.random.default_rng(rng)
        #self.current_loglike = self.loglike_fn(self.current_params)

    def __init__(self, like):
        """
        Initialize Metropolis sampler settings and storage arrays.

        Parameters
        ----------
        like : Likelihood
            Likelihood object used to evaluate log-likelihood values.

        Returns
        -------
        None
        """
        self.like = like
        self.n = 2500  #integration steps
        self.steps = 1000

        self.step_size = np.array([0.3, 0.07, 0.1])
        self.rng = np.random.default_rng()

        self.current_params = np.array([68, 0.15, 0.9])
        # Start away from best-fit so burn-in behavior is visible in the trace.

        self.current_loglike = -self.like.negLogLikelihood(self.current_params, self.n, model_type = "standard")

        self.chain = np.zeros((self.steps, 3))
        self.logL_chain = np.zeros(self.steps)

        self.chain[0] = self.current_params  #stores i=0
        self.logL_chain[0] = self.current_loglike
    
    def proposal(self):
        """
        Propose a new parameter point using a Gaussian random walk.

        Returns
        -------
        p_prime : np.ndarray
            Proposed parameter vector [H0, Omega_m, Omega_lambda].
        """
        step = self.rng.normal(0.0, scale = self.step_size, size = 3)
        # Independent Gaussian jump in each parameter dimension.
        p_prime = self.current_params + step
        return p_prime
    
    def run(self):
        """
        Execute the Metropolis sampling loop and record the full chain.

        Returns
        -------
        None
        """

        acceptance_count = 0

        for i in range(1, self.steps):
            p_prime = self.proposal()
            logL_prime = -self.like.negLogLikelihood(p_prime, self.n, model_type = "standard")

            if logL_prime > self.current_loglike:
                # Always accept uphill moves in log-likelihood.
                accept = True

            else:
                u = np.random.uniform(0.0, 1.0)
                accept = (np.log(u) < (logL_prime - self.current_loglike))

            if accept == True:
                self.current_params = p_prime
                self.current_loglike = logL_prime
            # If rejected, we intentionally keep previous state (pi+1 = pi).

                acceptance_count += 1  #tracks acceptance for acceptance rate calc


            self.chain[i] = self.current_params
            self.logL_chain[i] = self.current_loglike
            # Record every step, including repeats

        acceptance_rate = acceptance_count / self.steps
        print(f"Acceptance rate: {acceptance_rate*100}%")

        self.analysis()  #analysis and graphing

    def plot_loglike_trace(self):
        """
        Plot log-likelihood values against MCMC step number.

        Returns
        -------
        None
        """
        plt.figure()
        plt.plot(self.logL_chain)
        plt.xlabel("Step")
        plt.ylabel("log-likelihood")
        plt.title("Burn In period for Log-Likelihood")
        plt.ylim(-700, -500)
        plt.show()

    def analysis(self):
        """
        Perform burn-in removal and generate posterior diagnostic plots.

        Returns
        -------
        None
        """
        burn_in = 200
        burnin_params = self.chain[burn_in:]
        burnin_logL = self.logL_chain[burn_in:]
        # Burn-in removes initial transient while chain moves toward high-probability region.

        H0 = burnin_params[:, 0]
        Om = burnin_params[:, 1]
        Ol = burnin_params[:, 2]
        # Columns follow parameter order [H0, Omega_m, Omega_lambda].

        #1d probability distributions for each parameter
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.hist(H0, bins = 40)
        plt.xlabel("H0")
        plt.ylabel("Probability Distribution")

        plt.subplot(1, 3, 2)
        plt.hist(Om, bins = 40)
        plt.xlabel("Omega m")
        plt.ylabel("Probability Distribution")

        plt.subplot(1, 3, 3)
        plt.hist(Ol, bins = 40)
        plt.xlabel("Omega lambda")
        plt.ylabel("Probability Distribution")
        plt.show()


        #3d plots
        plt.figure()
        scat = plt.scatter(Om, H0, c=Ol)
        # Color maps the third parameter to give a 3D view in a 2D plot.
        plt.xlabel("Omega m")
        plt.ylabel("H0")
        plt.title("H0 against Omega m with Omega lambda represented with colour")
        plt.colorbar(scat, label = "Omega lambda")
        plt.show()

    def runMultipleChains(self, M, N, burn_in):

        chain_list = np.zeros((M, N - burn_in, 3))

        for m in range(M):
            metro = Metropolis(self.like)
            metro.n = self.n
            metro.steps = N
            metro.step_size = self.step_size.copy()

            metro.chain = np.zeros((metro.steps, 3))
            metro.logL_chain = np.zeros(metro.steps)

            metro.current_params = self.current_params
            metro.current_loglike = -metro.like.negLogLikelihood(metro.current_params, metro.n, model_type="standard")
            
            metro.chain[0] = metro.current_params
            metro.logL_chain[0] = metro.current_loglike

            metro.run()
            chain_list[m] = metro.chain[burn_in:]
        
        return chain_list
    
    def gelmanRubin(self):
        N = 1000  #length of chains
        M = 2  #number of chains
        burn_in = 200
        
        chains = self.runMultipleChains(M, N, burn_in)
        N_eff = N - burn_in  #number of elements in the chain after burn in

        chain_means = np.mean(chains, axis=1)
        chain_vars = np.var(chains, axis=1)

        B = np.var(chain_means, axis=0)
        W = np.mean(chain_vars, axis=0)

        R = np.sqrt(((N - 1) / N) + (((M + 1) / M) * (B / W)))
        print(f"Gelman Rubin statistic: {R}")

def main():
    """
    Set up cosmology objects and run the Metropolis workflow.

    Returns
    -------
    None
    """
    H0 = 70
    Omega_m = 0.3
    Omega_lambda = 0.7
    n = 2500 #number of integration steps

    cosmo = Cosmology(H0, Omega_m, Omega_lambda)
    like = Likelihood(cosmo)
    metro = Metropolis(like)


    metro.gelmanRubin()

    #standard model
    #best_params_std, best_fun_std = like.optimise(n, model_type = "standard")

    #no lambda model
    #best_params_nolambda, best_params_nolambda = like.optimise(n, model_type = "no_lambda")

    #likelihood grids
    #like_grid_3d, p0, p1, p2 = like.likelihoodGrid3d(n)
    #like.marginalizedLikelihoods(like_grid_3d, p0, p1, p2)

    metro.run()
    metro.plot_loglike_trace()

   
if __name__ == "__main__":
    main()