# Cosmological Parameter Estimation using MCMC

Python coursework for a Computer Modelling course. The six units build, step by step, a cosmological parameter-estimation pipeline: starting from a simple `Cosmology` class, adding numerical integration for cosmological distances, fitting a model to Type Ia supernova data, and finally sampling the posterior with a Metropolis–Hastings MCMC sampler.

The repo also contains a few small warm-up scripts from early in the course.

![Pantheon supernova data with best-fit model](Unit%204/Graphs/DistanceMod%20against%20redshift%20with%20best%20fit.png)

## What's covered

| Unit | Topic | Main file(s) |
|------|-------|--------------|
| 1 | Object-oriented basics: a `Cosmology` class holding H₀, Ω_m and Ω_Λ, deriving Ω_k, and plotting the Friedmann integrand against redshift | `unit1.py` |
| 2 | NumPy arrays, a Gaussian log-likelihood, and summary statistics (mean, standard deviation, 5th/95th percentiles) with histograms and scatter plots for a set of parameter samples | `unit2.py`, `data.txt` |
| 3 | Numerical integration of the comoving distance using the rectangle, trapezoid and Simpson rules; convergence study of fractional error vs. number of evaluations; cumulative trapezoid integration, interpolation and distance moduli | `unit3.py` |
| 4 | Fitting cosmological models to the Pantheon supernova sample: a `Likelihood` class, convergence of the log-likelihood with integration steps, and best-fit parameters via `scipy.optimize.minimize` for a ΛCDM model and a model with Ω_Λ = 0, plus residual analysis | `cosmology.py`, `unit4.py` |
| 5 | Parameter uncertainties: a 3D likelihood grid marginalised to 1D and 2D distributions, and a first Metropolis MCMC sampler with burn-in removal | `cosmology.py`, `unit5.py` |
| 6 | MCMC diagnostics: acceptance rate, mean and error on the mean for each parameter, running-mean convergence plots, multiple chains and the Gelman–Rubin statistic | `cosmology.py`, `likelihood.py`, `metropolis.py` |

Each unit folder also contains its brief (PDF), a results write-up, and the graphs produced.

### Other folders

- **Hypotenuse Finder** – reads two side lengths and prints the hypotenuse, with the calculation split into its own module.
- **Graphing Test** – plots y = x² with Matplotlib.
- **Fizzbuzz** – the classic FizzBuzz exercise.

## The physics

The core calculation is the luminosity distance in an FLRW universe. For each redshift *z* the code integrates

$$
D_C(z) = \frac{c}{H_0}\int_0^z \frac{dz'}{\sqrt{\Omega_m (1+z')^3 + \Omega_k (1+z')^2 + \Omega_\Lambda}}
$$

with Ω_k = 1 − Ω_m − Ω_Λ, converts it to a luminosity distance (using sinh or sin for open or closed geometries), and then to a distance modulus

$$
\mu = 5\log_{10}\left(\frac{D_L}{\text{Mpc}}\right) + 25.
$$

Model magnitudes (μ plus an absolute magnitude M = −19.3) are compared with the observed Pantheon values through a Gaussian log-likelihood

$$
\ln \mathcal{L} = -\frac{1}{2}\sum_i \frac{(\mu_{\text{obs},i} - m_i)^2}{\sigma_i^2}.
$$

This likelihood is then maximised (Unit 4), evaluated on a grid (Unit 5) and sampled with MCMC (Units 5 and 6).

## Requirements

- Python 3.11 (other recent 3.x versions should work)
- NumPy
- SciPy
- Matplotlib
- tabulate (Unit 2 only)

```bash
pip install numpy scipy matplotlib tabulate
```

## Running the code

Scripts load their data files with relative paths, so run each one from inside its own folder:

```bash
cd "Unit 4"
python unit4.py
```

```bash
cd "Unit 6"
python metropolis.py
```

Most scripts open Matplotlib windows one after another; close each window to move on to the next plot. In Units 5 and 6 several analyses (optimisation, likelihood grids, running means, Gelman–Rubin) are commented out in `main()` and can be switched on as needed.

Unit 6 runs a 16,000-step chain by default, with 2,500 integration steps per likelihood evaluation, so it takes a while to finish.

## Data

`pantheon_data.txt` (Units 4–6) contains redshift, observed distance modulus and its uncertainty for the Pantheon Type Ia supernova sample:

```
# z mu_obs mu_err
```

`Unit 2/data.txt` contains samples of Ω_m h², Ω_b h² and H₀.

## Example results

Output from one MCMC run (`Unit 5/Graphs/20000 20000 30 Graphs/stats.txt`):

| Parameter | Mean ± error on mean |
|-----------|----------------------|
| H₀ | 72.1330 ± 0.0022 |
| Ω_m | 0.3473 ± 0.0003 |
| Ω_Λ | 0.8252 ± 0.0005 |

Acceptance rate: 8.74%

Full discussion of the results is in the results documents in each unit folder.

![Integration error vs number of evaluations](Unit%203/Fractional%20error%20plot.png)

## Repository structure

```
Computer-Modelling/
├── Unit 1/            Cosmology class basics
├── Unit 2/            NumPy, statistics and plotting
├── Unit 3/            Numerical integration and distance moduli
├── Unit 4/            Likelihood and best-fit optimisation
├── Unit 5/            Likelihood grids and first MCMC sampler
├── Unit 6/            MCMC convergence diagnostics
├── Hypotenuse Finder/
├── Graphing Test/
└── Fizzbuzz/
```

## Author

Aaron Liddell
