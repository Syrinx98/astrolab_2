"""
12 Building the Log-Likelihood and Running MCMC to Determine Posteriors

In this code, we:

1. Define a log-likelihood function for our transit model considering TESS and TASTE data.
2. Include limb darkening and polynomial trend parameters into the model.
3. Add jitter parameters for both TESS and TASTE datasets to handle unknown additional noise.
4. Define priors for some parameters (limb darkening in this example) using normal distributions.
5. Set boundaries for all parameters to avoid exploring unphysical values.
6. Combine log_likelihood and log_prior into a log_probability function for MCMC.
7. Use emcee (an affine invariant MCMC sampler) to sample from the posterior.
8. Save and analyze the results (trace plots, corner plots, and parameter summaries).
9. Optionally, show how to convert from scaled parameters (Rp/Rs) to physical units with stellar radius and error propagation.

Note:
- The code uses some data loaded from pickle files that must be generated beforehand.
- The code is adapted from the Moodle text. Make sure to have installed `emcee`, `batman-package`, `wotan`, `scipy`, and `matplotlib`.
- Adjust file names and parameters as needed for your specific target and data.

We assume:
- The TESS and TASTE data, as well as the modeling steps, follow from previous parts of the analysis.
- The arrays `tess_bjd_tdb`, `tess_normalized_flux`, `tess_normalized_ferr`, `taste_bjd_tdb`,
  `differential_allref`, `differential_allref_error` are already loaded.
- The initial guess `theta`, boundaries, and priors as per the text have been set.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import batman
from scipy import stats
import emcee
import corner
from multiprocessing import Pool

# =============================================================================
# 9.1. Loading Data
# =============================================================================

print("\n\n 9.1. Loading Data")
print("=============================================================================\n")

# Define directories (update these paths as necessary)
taste_dir = "TASTE_analysis/group05_QATAR-1_20230212"
tess_dir = "TESS_analysis"

# Load TASTE data
taste_bjd_tdb = pickle.load(open(f'{taste_dir}/taste_bjdtdb.p', 'rb'))
differential_allref = pickle.load(open(f'{taste_dir}/differential_allref.p', 'rb'))
differential_allref_error = pickle.load(open(f'{taste_dir}/differential_allref_error.p', 'rb'))

# Load TESS data (sector 24 as an example)
tess_sector_dict = pickle.load(open(f'{tess_dir}/qatar1_TESS_sector024_filtered.p', 'rb'))
tess_bjd_tdb = tess_sector_dict['time']
tess_normalized_flux = tess_sector_dict['selected_flux']
tess_normalized_ferr = tess_sector_dict['selected_flux_error']

print("Data loaded successfully.")

# =============================================================================
# 9.2. Defining Parameters and Model Setup
# =============================================================================

print("\n\n 9.2. Defining Parameters and Model Setup")
print("=============================================================================\n")

"""
We have 14 parameters indexed from 0 to 13:
  0: t0   (time of inferior conjunction, BJD_TDB)
  1: per  (orbital period in days)
  2: rp   (planet-to-star radius ratio, Rp/Rs)
  3: a    (semi-major axis in stellar radii, a/Rs)
  4: inc  (inclination in degrees)
  5,6: TESS limb darkening coefficients [u1, u2]
  7,8: TASTE limb darkening coefficients [u1, u2]
  9,10,11: polynomial trend for TASTE (0th, 1st, 2nd order)
  12: jitter for TESS
  13: jitter for TASTE

Below we provide an initial guess for these parameters. 
Here, we've updated them to reflect the typical values found earlier for 
Qatar-1b and the corresponding limb darkening from TESS and TASTE analyses.
"""

theta_initial = np.empty(14)

# -- Orbital/Transit parameters (for Qatar-1b as an example) --
theta_initial[0] = 2457475.204489  # t0 (mid-transit, BJD_TDB)
theta_initial[1] = 1.420024443     # per (days)
theta_initial[2] = 0.1463          # rp = Rp/Rs
theta_initial[3] = 6.25            # a = a/Rs
theta_initial[4] = 84.08           # inc (degrees)

# -- Limb darkening coefficients (updated from previous analysis) --
# TESS (u1, u2) -> [0.51, 0.10]
theta_initial[5] = 0.51
theta_initial[6] = 0.10
# TASTE (u1, u2) -> [0.65, 0.08]
theta_initial[7] = 0.65
theta_initial[8] = 0.08

# -- Polynomial trend for TASTE (initial guesses for 0th, 1st, 2nd order) --
theta_initial[9] = 0.245
theta_initial[10] = 0.0
theta_initial[11] = 0.0

# -- Jitter terms for TESS and TASTE --
theta_initial[12] = 0.0
theta_initial[13] = 0.0

# =============================================================================
# 9.3. Defining the log_likelihood function
# =============================================================================

print("\n\n 9.3. Defining the log_likelihood function")
print("=============================================================================\n")


def log_likelihood(theta):
    """
    Computes the log-likelihood for the given set of parameters (theta).
    We create TransitParams for TESS, then build a batman transit model.
    For TASTE, we use separate limb darkening but the same orbital geometry.
    We also multiply the TASTE model by a polynomial trend in time.
    Finally, we include per-dataset jitter (theta[12] for TESS, theta[13] for TASTE),
    and compute the Gaussian log-likelihood.
    """
    # Unpack parameters
    t0, per, rp, a, inc = theta[0:5]
    tess_u1, tess_u2 = theta[5], theta[6]
    taste_u1, taste_u2 = theta[7], theta[8]
    poly0, poly1, poly2 = theta[9], theta[10], theta[11]
    tess_jitter, taste_jitter = theta[12], theta[13]

    # Create the batman.TransitParams object
    params = batman.TransitParams()
    params.t0 = t0
    params.per = per
    params.rp = rp
    params.a = a
    params.inc = inc
    params.ecc = 0.0
    params.w = 90.0
    params.limb_dark = "quadratic"

    # -- Model for TESS --
    params.u = [tess_u1, tess_u2]
    m_tess = batman.TransitModel(params, tess_bjd_tdb)
    tess_model_flux = m_tess.light_curve(params)

    # -- Model for TASTE --
    # We switch the limb darkening to the TASTE coefficients
    params.u = [taste_u1, taste_u2]
    median_bjd = np.median(taste_bjd_tdb)
    poly_trend = poly0 + poly1 * (taste_bjd_tdb - median_bjd) + poly2 * (taste_bjd_tdb - median_bjd) ** 2
    m_taste = batman.TransitModel(params, taste_bjd_tdb)
    taste_model_flux = m_taste.light_curve(params) * poly_trend

    # -- Add jitter in quadrature for TESS and TASTE --
    tess_errors_with_jitter = tess_normalized_ferr ** 2 + tess_jitter ** 2
    taste_errors_with_jitter = differential_allref_error ** 2 + taste_jitter ** 2

    # -- Compute chi-square for TESS and TASTE separately --
    chi2_tess = np.sum((tess_normalized_flux - tess_model_flux) ** 2 / tess_errors_with_jitter)
    chi2_taste = np.sum((differential_allref - taste_model_flux) ** 2 / taste_errors_with_jitter)

    # -- Total number of data points for TESS + TASTE --
    N_tess = len(tess_normalized_flux)
    N_taste = len(differential_allref)
    N = N_tess + N_taste

    # -- Sum of log(error^2) for TESS and TASTE --
    sum_ln_sigma_tess = np.sum(np.log(tess_errors_with_jitter))
    sum_ln_sigma_taste = np.sum(np.log(taste_errors_with_jitter))

    # -- Gaussian log-likelihood expression --
    logL = -0.5 * (
        N * np.log(2.0 * np.pi)
        + chi2_tess + chi2_taste
        + sum_ln_sigma_tess + sum_ln_sigma_taste
    )

    return logL


# Quick test of log_likelihood at the initial guess
print("log_likelihood at initial guess:", log_likelihood(theta_initial))

# =============================================================================
# 9.4. Defining Priors
# =============================================================================

print("\n\n 9.4. Defining Priors")
print("=============================================================================\n")


def log_prior(theta):
    """
    Defines the log-prior for the parameters.
    We place Gaussian priors on the limb darkening coefficients only.
    """
    # Limb darkening priors
    tess_u1_mean, tess_u1_std = 0.51, 0.05
    tess_u2_mean, tess_u2_std = 0.10, 0.05
    taste_u1_mean, taste_u1_std = 0.65, 0.05
    taste_u2_mean, taste_u2_std = 0.08, 0.05

    # Initialize log prior
    lp = 0.0

    # TESS limb darkening coefficients
    lp += np.log(stats.norm.pdf(theta[5], loc=tess_u1_mean, scale=tess_u1_std))
    lp += np.log(stats.norm.pdf(theta[6], loc=tess_u2_mean, scale=tess_u2_std))

    # TASTE limb darkening coefficients
    lp += np.log(stats.norm.pdf(theta[7], loc=taste_u1_mean, scale=taste_u1_std))
    lp += np.log(stats.norm.pdf(theta[8], loc=taste_u2_mean, scale=taste_u2_std))

    return lp


# =============================================================================
# 9.5. Defining Boundaries and log_probability
# =============================================================================

print("\n\n 9.5. Defining Boundaries and log_probability")
print("=============================================================================\n")

# Define boundaries for parameters (lower and upper bounds)
# Shape: (2, 14) -> row0 = lower bounds, row1 = upper bounds
boundaries = np.empty((2, 14))

# Parameter 0: t0 (mid-transit time) [t0_initial - 0.5, t0_initial + 0.5]
boundaries[:, 0] = [theta_initial[0] - 0.5, theta_initial[0] + 0.5]

# Parameter 1: per (period) [per_initial - 0.5, per_initial + 0.5]
boundaries[:, 1] = [theta_initial[1] - 0.5, theta_initial[1] + 0.5]

# Parameter 2: rp (Rp/Rs) [0.0, 0.5]
boundaries[:, 2] = [0.0, 0.5]

# Parameter 3: a (a/Rs) [0.0, 20.0]
boundaries[:, 3] = [0.0, 20.0]

# Parameter 4: inc (inclination) [0.0, 90.0]
boundaries[:, 4] = [0.0, 90.0]

# Parameters 5 & 6: TESS limb darkening coefficients [u1, u2] [0.0, 1.0]
boundaries[:, 5] = [0.0, 1.0]
boundaries[:, 6] = [0.0, 1.0]

# Parameters 7 & 8: TASTE limb darkening coefficients [u1, u2] [0.0, 1.0]
boundaries[:, 7] = [0.0, 1.0]
boundaries[:, 8] = [0.0, 1.0]

# Parameters 9, 10, 11: Polynomial trend for TASTE
boundaries[:, 9] = [0.0, 1.0]    # poly0
boundaries[:, 10] = [-1.0, 1.0]  # poly1
boundaries[:, 11] = [-1.0, 1.0]  # poly2

# Parameters 12 & 13: Jitter terms for TESS and TASTE [0.0, 0.05]
boundaries[:, 12] = [0.0, 0.05]
boundaries[:, 13] = [0.0, 0.05]


def log_probability(theta):
    """
    Combines log_prior and log_likelihood.
    Returns -infinity if any parameter is outside its boundaries or prior is -infinity.
    """
    # Check if any parameter is outside its boundaries
    if np.any(theta < boundaries[0, :]) or np.any(theta > boundaries[1, :]):
        return -np.inf

    # Compute log_prior
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf

    # Compute log_likelihood
    ll = log_likelihood(theta)
    if not np.isfinite(ll):
        return -np.inf

    return lp + ll


print("log_probability at initial guess:", log_probability(theta_initial))

# =============================================================================
# 9.6. Running the MCMC Sampler with emcee
# =============================================================================

print("\n\n 9.6. Running the MCMC Sampler with emcee")
print("=============================================================================\n")


def run_mcmc(theta_initial, boundaries, nwalkers=50, nsteps=20000, filename=f'{tess_dir}/emcee_sampler_first_run.p'):
    """
    Runs the MCMC sampler using emcee and saves the sampler object to a file.
    """
    ndim = len(theta_initial)

    # Initialize walkers: small Gaussian perturbation around the initial guess
    starting_point = theta_initial + 1e-5 * np.random.randn(nwalkers, ndim)

    # Set up the multiprocessing Pool
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
        sampler.run_mcmc(starting_point, nsteps, progress=True)

    # Save the sampler object for later analysis
    with open(filename, 'wb') as f:
        pickle.dump(sampler, f)

    print(f"MCMC run completed and results saved to '{filename}'.")


# Ensure that the MCMC run is protected under __main__ to avoid multiprocessing issues
if __name__ == "__main__":
    tess_dir = "TESS_analysis"

    run_mcmc(theta_initial, boundaries)

    # =============================================================================
    # 9.7. Analyzing the MCMC Results
    # =============================================================================

    print("\n\n 9.7. Analyzing the MCMC Results")
    print("=============================================================================\n")

    # Load the sampler object
    with open(f'{tess_dir}/emcee_sampler_first_run.p', 'rb') as f:
        sampler = pickle.load(f)

    # Discard burn-in steps and thin the chain
    burn_in = 2500
    thinning = 100
    flat_samples = sampler.get_chain(discard=burn_in, thin=thinning, flat=True)
    print("Flat samples shape:", flat_samples.shape)

    # Parameter summary: 16th, 50th, 84th percentiles
    param_names = [
        "t0", "per", "rp", "a", "inc",
        "TESS_u1", "TESS_u2", "TASTE_u1", "TASTE_u2",
        "poly0", "poly1", "poly2",
        "jitter_TESS", "jitter_TASTE"
    ]

    for i in range(len(theta_initial)):
        mcmc = np.percentile(flat_samples[:, i], [15.865, 50.0, 84.135])
        q = np.diff(mcmc)
        print(f"{param_names[i]}: {mcmc[1]:.7f} -{q[0]:.7f} +{q[1]:.7f}")

    # Corner plot of the posterior distributions
    fig = corner.corner(
        flat_samples,
        labels=param_names,
        show_titles=True,
        quantiles=[0.16, 0.5, 0.84],
        title_fmt=".5f",
        title_kwargs={"fontsize": 12}
    )
    plt.show()

    # =============================================================================
    # 9.8. Extract Best-Fit Model and Compare with Data
    # =============================================================================

    print("\n\n 9.8. Extract Best-Fit Model and Compare with Data")
    print("=============================================================================\n")

    # Take the median of each parameter as the "best-fit"
    theta_best = np.median(flat_samples, axis=0)
    print("Best-fit parameters (median of posterior):", theta_best)

    # Rebuild the batman model at the best-fit
    params_best = batman.TransitParams()
    params_best.t0 = theta_best[0]
    params_best.per = theta_best[1]
    params_best.rp = theta_best[2]
    params_best.a = theta_best[3]
    params_best.inc = theta_best[4]
    params_best.ecc = 0.0
    params_best.w = 90.0
    params_best.limb_dark = "quadratic"

    # -- Model for TESS --
    params_best.u = [theta_best[5], theta_best[6]]
    m_tess_best = batman.TransitModel(params_best, tess_bjd_tdb)
    tess_flux_model_best = m_tess_best.light_curve(params_best)

    # -- Model for TASTE --
    params_best.u = [theta_best[7], theta_best[8]]
    median_bjd = np.median(taste_bjd_tdb)
    poly_trend_best = (
        theta_best[9]
        + theta_best[10] * (taste_bjd_tdb - median_bjd)
        + theta_best[11] * (taste_bjd_tdb - median_bjd) ** 2
    )
    m_taste_best = batman.TransitModel(params_best, taste_bjd_tdb)
    taste_flux_model_best = m_taste_best.light_curve(params_best) * poly_trend_best

    # Plot TASTE data vs best-fit model
    plt.figure(figsize=(10, 6))
    plt.scatter(taste_bjd_tdb, differential_allref, s=2, label='TASTE data', color='black')
    plt.plot(taste_bjd_tdb, taste_flux_model_best, lw=2, c='C1', label='Best-fit model')
    plt.xlabel("BJD TDB")
    plt.ylabel("Relative flux")
    plt.legend()
    plt.title("TASTE Data with Best-Fit Model")
    plt.show()

    # Plot TESS data vs best-fit model
    plt.figure(figsize=(10, 6))
    plt.scatter(tess_bjd_tdb, tess_normalized_flux, s=2, label='TESS data', color='black')
    plt.plot(tess_bjd_tdb, tess_flux_model_best, lw=2, c='C1', label='Best-fit model')
    plt.xlabel("BJD TDB")
    plt.ylabel("Relative flux")
    plt.legend()
    plt.title("TESS Data with Best-Fit Model")
    plt.show()

    # =============================================================================
    # 9.9. Deriving Physical Parameters (Example: Planet Radius)
    # =============================================================================

    print("\n\n 9.9. Deriving Physical Parameters (Example: Planet Radius)")
    print("=============================================================================\n")

    """
    Example of converting the fitted Rp/Rs into a physical radius in Jupiter radii.
    We assume (and sample) a stellar radius distribution. 
    Then, Rp [Rjup] = (Rp/Rs) * R_star [R_sun] * (R_sun / R_jup).
    """

    Rp_Rs_samples = flat_samples[:, 2]  # the chain for rp = Rp/Rs
    n_samples = len(Rp_Rs_samples)

    # Suppose we have a prior on stellar radius (e.g., R_star = 0.48 +/- 0.04 R_sun)
    # Adjust these values based on your stellar parameter estimates
    r_star_mean = 0.48  # in R_sun
    r_star_std = 0.04   # in R_sun
    r_star_samples = np.random.normal(r_star_mean, r_star_std, n_samples)  # R_sun

    # Constants for conversion
    R_sun_km = 695700.0  # km
    R_jup_km = 71492.0    # km
    factor_sun_to_jup = R_sun_km / R_jup_km  # approximately 9.73

    # Compute planet radius in R_jup
    Rp_jup = r_star_samples * Rp_Rs_samples * factor_sun_to_jup

    # Summary statistics
    Rp_mean = np.mean(Rp_jup)
    Rp_std = np.std(Rp_jup)
    print(f"Planet radius [Rjup]: mean = {Rp_mean:.6f}, std = {Rp_std:.6f}")

    print("All steps completed successfully.")
