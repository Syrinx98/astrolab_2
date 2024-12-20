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
from astropy.io import fits
import pickle
import batman
from scipy import stats
import emcee
import corner

# =============================================================================
print("\n\n 9.1. Loading Data")
print("=============================================================================\n")

taste_bjd_tdb = pickle.load(open('taste_bjdtdb.p','rb'))
differential_allref = pickle.load(open('differential_allref.p','rb'))
differential_allref_error = pickle.load(open('differential_allref_error.p','rb'))

tess_sector44_dict = pickle.load(open('GJ3470_TESS_sector044_filtered.p','rb'))
tess_bjd_tdb = tess_sector44_dict['time']
tess_normalized_flux = tess_sector44_dict['selected_flux']
tess_normalized_ferr = tess_sector44_dict['selected_flux_error']

print("Data loaded successfully.")


# =============================================================================
print("\n\n 9.2. Defining Parameters and Model Setup")
print("=============================================================================\n")

# According to the instructions, we have 14 parameters indexed from 0 to 13:
# 0: t0 (time of inferior conjunction)
# 1: per (period)
# 2: rp (planet radius in stellar radii)
# 3: a (semi-major axis in stellar radii)
# 4: inc (inclination in degrees)
# 5,6: TESS limb darkening u1, u2
# 7,8: TASTE limb darkening u1, u2
# 9,10,11: polynomial trend for TASTE (0-th, 1st, 2nd order)
# 12: jitter for TESS
# 13: jitter for TASTE

theta = np.empty(14)
theta[0] = 2459500.53574
theta[1] = 3.33665
theta[2] = 0.0764
theta[3] = 13.94
theta[4] = 88.9
theta[5] = 0.35
theta[6] = 0.23
theta[7] = 0.58
theta[8] = 0.18
theta[9] = 0.245
theta[10]= 0.0
theta[11]= 0.0
theta[12]= 0.0
theta[13]= 0.0

# Transit parameters required by batman
# We'll define them in the log_likelihood function dynamically.


# =============================================================================
print("\n\n 9.3. Defining the log_likelihood function")
print("=============================================================================\n")

def log_likelihood(theta):

    params = batman.TransitParams()
    params.t0 =  theta[0]
    params.per = theta[1]
    params.rp =  theta[2]
    params.a =   theta[3]
    params.inc = theta[4]
    params.ecc = 0.
    params.w = 90.
    params.u = [theta[5], theta[6]]
    params.limb_dark = "quadratic"

    # TESS model
    m_tess = batman.TransitModel(params, tess_bjd_tdb)
    tess_model_flux = m_tess.light_curve(params)

    # TASTE model
    params.u = [theta[7], theta[8]]
    median_bjd = np.median(taste_bjd_tdb)
    polynomial_trend = theta[9] + theta[10]*(taste_bjd_tdb - median_bjd) + theta[11]*(taste_bjd_tdb - median_bjd)**2
    m_taste = batman.TransitModel(params, taste_bjd_tdb)
    taste_model_flux = m_taste.light_curve(params) * polynomial_trend

    # Add jitter in quadrature
    tess_errors_with_jitter = tess_normalized_ferr**2 + theta[12]**2
    taste_errors_with_jitter = differential_allref_error**2 + theta[13]**2

    # Compute chi2
    chi2_tess = np.sum((tess_normalized_flux - tess_model_flux)**2 / tess_errors_with_jitter)
    chi2_taste = np.sum((differential_allref - taste_model_flux)**2 / taste_errors_with_jitter)

    N = len(tess_errors_with_jitter) + len(taste_errors_with_jitter)
    sum_ln_sigma_tess = np.sum(np.log(tess_errors_with_jitter))
    sum_ln_sigma_taste = np.sum(np.log(taste_errors_with_jitter))

    logL = -0.5 * ( N*np.log(2*np.pi) + chi2_tess + chi2_taste + sum_ln_sigma_tess + sum_ln_sigma_taste)
    return logL


# Test log_likelihood with initial theta
print("log_likelihood at initial guess:", log_likelihood(theta))


# =============================================================================
print("\n\n 9.4. Defining Priors")
print("=============================================================================\n")

# Priors on limb darkening coefficients (example)
# Using normal distributions with given mean and std.
# TESS: u1 = 0.35±0.10, u2=0.23±0.10
# TASTE: u1=0.58±0.10, u2=0.18±0.10

def log_prior(theta):
    prior = 0.0
    # TESS LD priors
    prior += np.log(stats.norm.pdf(theta[5], loc=0.35, scale=0.10))
    prior += np.log(stats.norm.pdf(theta[6], loc=0.23, scale=0.10))
    # TASTE LD priors
    prior += np.log(stats.norm.pdf(theta[7], loc=0.58, scale=0.10))
    prior += np.log(stats.norm.pdf(theta[8], loc=0.18, scale=0.10))
    return prior


# =============================================================================
print("\n\n 9.5. Defining Boundaries and log_probability")
print("=============================================================================\n")

boundaries = np.empty([2, len(theta)])
boundaries[:,0] = [theta[0]-0.5, theta[0]+0.5]
boundaries[:,1] = [theta[1]-0.5, theta[1]+0.5]
boundaries[:,2] = [0.0, 0.5]
boundaries[:,3] = [0.0, 20.]
boundaries[:,4] = [0.00, 90.0]
boundaries[:,5] = [0.00, 1.0]
boundaries[:,6] = [0.00, 1.0]
boundaries[:,7] = [0.00, 1.0]
boundaries[:,8] = [0.00, 1.0]
boundaries[:,9] = [0.00, 1.0]
boundaries[:,10]= [-1.0, 1.0]
boundaries[:,11]= [-1.0, 1.0]
boundaries[:,12]= [0.0, 0.05]
boundaries[:,13]= [0.0, 0.05]

def log_probability(theta):
    sel = (theta < boundaries[0,:]) | (theta > boundaries[1,:])
    if np.sum(sel) > 0:
        return -np.inf
    lp = log_prior(theta)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(theta)

print("log_probability at initial guess:", log_probability(theta))


# =============================================================================
print("\n\n 9.6. Running the MCMC Sampler with emcee")
print("=============================================================================\n")

nwalkers = 50
nsteps = 20000
ndim = len(theta)

# Initialize walkers around the initial guess
starting_point = theta + np.abs(1e-5 * np.random.randn(nwalkers, ndim))

from multiprocessing import Pool
with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
    sampler.run_mcmc(starting_point, nsteps, progress=True)

pickle.dump(sampler, open('emcee_sampler_first_run.p','wb'))
print("MCMC run completed and saved.")


# =============================================================================
print("\n\n 9.7. Analyzing the MCMC Results")
print("=============================================================================\n")

sampler = pickle.load(open('emcee_sampler_first_run.p','rb'))
flat_samples = sampler.get_chain(discard=2500, thin=100, flat=True)
print("Flat samples shape:", flat_samples.shape)

# Parameter summary
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [15.865, 50, 84.135])
    q = np.diff(mcmc)
    print("Param {0}: {1:.7f} -{2:.7f} +{3:.7f}".format(i, mcmc[1], q[0], q[1]))


# Corner plot
fig = corner.corner(flat_samples, labels=[f"theta_{i}" for i in range(ndim)])
plt.show()


# =============================================================================
print("\n\n 9.8. Extract Best-Fit Model and Compare with Data")
print("=============================================================================\n")

theta_best = np.median(flat_samples, axis=0)

params = batman.TransitParams()
params.t0 = theta_best[0]
params.per = theta_best[1]
params.rp = theta_best[2]
params.a = theta_best[3]
params.inc = theta_best[4]
params.ecc = 0.
params.w = 90.
params.u = [theta_best[5], theta_best[6]]
params.limb_dark = "quadratic"

m_tess = batman.TransitModel(params, tess_bjd_tdb)
tess_flux_model = m_tess.light_curve(params)

params.u = [theta_best[7], theta_best[8]]
median_bjd = np.median(taste_bjd_tdb)
poly_trend = theta_best[9] + theta_best[10]*(taste_bjd_tdb - median_bjd) + theta_best[11]*(taste_bjd_tdb - median_bjd)**2
m_taste = batman.TransitModel(params, taste_bjd_tdb)
taste_flux_model = m_taste.light_curve(params)*poly_trend

plt.figure(figsize=(6,4))
plt.scatter(taste_bjd_tdb, differential_allref, s=2)
plt.plot(taste_bjd_tdb, taste_flux_model, lw=2, c='C1')
plt.xlabel("BJD TDB")
plt.ylabel("Relative flux")
plt.title("TASTE Data with Best-Fit Model")
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(tess_bjd_tdb, tess_normalized_flux, s=2)
plt.plot(tess_bjd_tdb, tess_flux_model, lw=2, c='C1')
plt.xlabel("BJD TDB")
plt.ylabel("Relative flux")
plt.title("TESS Data with Best-Fit Model")
plt.show()


# =============================================================================
print("\n\n 9.9. Deriving Physical Parameters (Example: Planet Radius)")
print("=============================================================================\n")

# Example of converting Rp/Rs to Jupiter radii given stellar radius
Rp_Rs = flat_samples[:, 2]
n_samples = len(Rp_Rs)
r_star = np.random.normal(0.48, 0.04, n_samples)  # stellar radius in solar radii
# Radius of Sun = 695700 km, Radius of Jupiter = 71492 km
Rp = r_star * Rp_Rs * (695700 / 71492)

print("Planet radius [Rjup]: mean={0:.6f}, std={1:.6f}".format(np.mean(Rp), np.std(Rp)))
print("All steps completed.")
