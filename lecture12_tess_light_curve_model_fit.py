#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCMC Transit Fitting Script for Qatar-1b

This script performs the following steps:
1) Loads TASTE and TESS light curve data
2) Initializes model parameters and boundaries
3) Defines transit+trend model, likelihood, priors, and posterior
4) Runs emcee to sample the posterior distribution
5) Produces trace plots, corner plots, and best-fit comparison plots
6) Computes derived physical parameters (planet radius in Jupiter radii)
7) (Optional) Shows how to run a differential evolution optimizer for a starting point

Keep file paths consistent with the author's directory structure.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import batman
from scipy import stats
import emcee
import corner
from multiprocessing import Pool, freeze_support
from functools import partial
import warnings

# =============================================================================
# 1) Data Loading -----------------------------------------------------------
def load_data():
    """
    Load TASTE and TESS photometric data from pickle files.
    Normalize both light curves to unit out-of-transit median.
    Returns arrays for times, fluxes, errors, and directories.
    """
    print("[1] Loading data...")
    taste_dir = 'TASTE_analysis/group05_QATAR-1_20230212'
    tess_dir = 'TESS_analysis'

    # Load TASTE data
    taste_bjd = pickle.load(open(f'{taste_dir}/taste_bjdtdb.p', 'rb'))
    differential_allref = pickle.load(open(f'{taste_dir}/differential_allref.p', 'rb'))
    differential_allref_error = pickle.load(open(f'{taste_dir}/differential_allref_error.p', 'rb'))
    # Normalize TASTE
    median_taste = np.median(differential_allref)
    differential_allref /= median_taste
    differential_allref_error /= median_taste

    # Load TESS data
    tess_dict = pickle.load(open(f'{tess_dir}/qatar1_TESS_sector024_filtered.p', 'rb'))
    tess_bjd = tess_dict['time']
    if 'pdcsap_flat' in tess_dict:
        tess_flux = tess_dict['pdcsap_flat']
        tess_flux_err = tess_dict['pdcsap_err']
    else:
        tess_flux = tess_dict['selected_flux']
        tess_flux_err = tess_dict['selected_flux_error']
    # Normalize TESS
    median_tess = np.median(tess_flux)
    tess_flux /= median_tess
    tess_flux_err /= median_tess

    print(f"  - TASTE: {len(taste_bjd)} points (normalized)")
    print(f"  - TESS: {len(tess_bjd)} points (normalized)")
    print("Data loaded and normalized successfully.")

    return (taste_bjd, differential_allref, differential_allref_error,
            tess_bjd, tess_flux, tess_flux_err,
            taste_dir, tess_dir)

# =============================================================================
# 2) Parameter and Boundary Initialization ---------------------------------
def init_theta():
    """
    Provide an initial guess for the 14 model parameters:
      0 t0, 1 per, 2 rp/Rs, 3 a/Rs, 4 inc,
      5-6 TESS u1, u2; 7-8 TASTE u1, u2;
      9-11 polynomial trend coeffs; 12-13 jitters.
    """
    print("[2] Initializing parameter guess...")
    theta0 = np.array([
        2457475.204489,  # t0 (BJD_TDB)
        1.420024443,     # per (days)
        0.1463,          # rp/Rs
        6.25,            # a/Rs
        84.08,           # inc (deg)
        # Refined from empirical analysis:
        0.515, 0.098,    # TESS limb darkening u1,u2
        0.652, 0.078,    # TASTE (SDSS r) u1,u2
        1.0, 0.0, 0.0,   # poly0, poly1, poly2 (trend)
        0.001, 0.001     # jitter_TESS, jitter_TASTE
    ])
    print("Initial theta:", theta0)
    return theta0


def init_boundaries(theta0):
    """
    Define physical boundaries for each parameter to avoid unphysical values.
    Boundaries format: boundaries[0,i] = lower bound, boundaries[1,i] = upper bound.
    """
    print("[3] Setting parameter boundaries...")
    npar = len(theta0)
    bounds = np.zeros((2, npar))
    # t0 within ±0.5 days
    bounds[0,0], bounds[1,0] = theta0[0] - 0.5, theta0[0] + 0.5
    # period within ±0.5 days
    bounds[0,1], bounds[1,1] = theta0[1] - 0.5, theta0[1] + 0.5
    # rp/Rs: physical range
    bounds[0,2], bounds[1,2] = 0.0, 0.5
    # a/Rs: physical range
    bounds[0,3], bounds[1,3] = 0.0, 20.0
    # inclination: [0,90]
    bounds[0,4], bounds[1,4] = 0.0, 90.0
    # limb darkening coefficients (TESS & TASTE): [0,1] for u1,u2
    bounds[0,5:9] = 0.0
    bounds[1,5:9] = 1.0
    # polynomial trend coefficients: poly0 [0,1], poly1 & poly2 [-1,1]
    bounds[0,9], bounds[1,9]   = 0.0, 1.0
    bounds[0,10], bounds[1,10] = -1.0, 1.0
    bounds[0,11], bounds[1,11] = -1.0, 1.0
    # jitters: small positive
    bounds[0,12], bounds[1,12] = 0.0, 0.05
    bounds[0,13], bounds[1,13] = 0.0, 0.05
    print("Boundaries set.")
    return bounds

# =============================================================================
# 3) Likelihood, Prior, and Posterior ---------------------------------------

def log_likelihood(theta, taste_bjd, diff, diff_err, tess_bjd, tess_flux, tess_flux_err):
    """
    Compute the Gaussian log-likelihood combining TESS and TASTE datasets,
    including jitter terms added in quadrature to measurement errors.
    """
    # Unpack parameters
    t0, per, rp, a, inc = theta[:5]
    tess_u1, tess_u2   = theta[5], theta[6]
    taste_u1, taste_u2 = theta[7], theta[8]
    poly0, poly1, poly2= theta[9:12]
    tess_jit, taste_jit= theta[12], theta[13]

    # Set up batman transit parameters
    params = batman.TransitParams()
    params.t0, params.per = t0, per
    params.rp, params.a, params.inc = rp, a, inc
    params.ecc, params.w = 0.0, 90.0
    params.limb_dark = 'quadratic'

    # TESS model light curve
    params.u = [tess_u1, tess_u2]
    m_tess = batman.TransitModel(params, tess_bjd)
    model_tess = m_tess.light_curve(params)

    # TASTE model + polynomial trend
    params.u = [taste_u1, taste_u2]
    m_taste = batman.TransitModel(params, taste_bjd)
    base_taste = m_taste.light_curve(params)
    median_bjd = np.median(taste_bjd)
    trend = poly0 + poly1*(taste_bjd - median_bjd) + poly2*(taste_bjd - median_bjd)**2
    model_taste = base_taste * trend

    # Compute residual variances (with jitter)
    err2_tess  = tess_flux_err**2  + tess_jit**2
    err2_taste = diff_err**2       + taste_jit**2

    # Chi-squared terms
    chi2_tess  = np.sum((tess_flux - model_tess)**2 / err2_tess)
    chi2_taste = np.sum((diff - model_taste)**2 / err2_taste)
    ln_det = np.sum(np.log(err2_tess)) + np.sum(np.log(err2_taste))
    Ntot = len(tess_flux) + len(diff)

    # Full log-likelihood
    ll = -0.5 * (Ntot * np.log(2 * np.pi) + chi2_tess + chi2_taste + ln_det)
    return ll


def log_prior(theta):
    """
    Gaussian priors on the four limb darkening coefficients with specified means and stds.
    All other parameters have uniform (flat) priors within their boundaries.
    """
    # Updated priors from lesson PDF:
    # TESS: u1=0.35±0.10, u2=0.23±0.10
    # TASTE: u1=0.58±0.05, u2=0.18±0.10
    u_means = [0.515, 0.098, 0.652, 0.078]  # Empirical values from lecture11
    u_stds  = [0.10, 0.10, 0.05, 0.10]
    lp = 0.0
    for i in range(4):
        lp += stats.norm.logpdf(theta[5+i], loc=u_means[i], scale=u_stds[i])
    return lp


def log_probability(theta, taste_bjd, diff, diff_err,
                    tess_bjd, tess_flux, tess_flux_err,
                    boundaries):
    """
    Computes log-prior + log-likelihood if theta within boundaries; otherwise -inf.
    """
    # Boundary check
    lower, upper = boundaries
    if np.any(theta < lower) or np.any(theta > upper):
        return -np.inf

    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf

    ll = log_likelihood(theta, taste_bjd, diff, diff_err,
                        tess_bjd, tess_flux, tess_flux_err)
    if not np.isfinite(ll):
        return -np.inf

    return lp + ll

# =============================================================================
# 4) Run MCMC ----------------------------------------------------------------
def run_mcmc(theta0, taste_bjd, diff, diff_err,
             tess_bjd, tess_flux, tess_flux_err,
             tess_dir, boundaries):
    """
    Set up and run the emcee EnsembleSampler, saving the sampler object.
    """
    print("\n[4] Running MCMC with emcee...")
    ndim = theta0.size
    nwalkers = 50  # ≥ 2×ndim
    nsteps   = 5000

    # Initialize walkers in a small ball around theta0
    p0 = theta0 + 1e-4 * np.random.randn(nwalkers, ndim)

    # Partial function to pass fixed data and boundaries
    prob_fn = partial(log_probability,
                      taste_bjd=taste_bjd, diff=diff, diff_err=diff_err,
                      tess_bjd=tess_bjd, tess_flux=tess_flux, tess_flux_err=tess_flux_err,
                      boundaries=boundaries)

    # Use multiprocessing for speed
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, prob_fn, pool=pool)
        print("Starting MCMC sampling...")
        sampler.run_mcmc(p0, nsteps, progress=True)
    
    # Extra diagnostic prints for chain behavior
    chain = sampler.get_chain()
    log_probs = sampler.get_log_prob()
    print("MCMC diagnostics:")
    print(f"  - Chain shape: {chain.shape}")
    print(f"  - Initial log-prob mean: {np.mean(log_probs[0]):.3f} ± {np.std(log_probs[0]):.3f}")
    print(f"  - Final log-prob mean: {np.mean(log_probs[-1]):.3f} ± {np.std(log_probs[-1]):.3f}")
    for i in range(ndim):
        param_chain = chain[:,:,i]
        print(f"Parameter {i}: mean={np.mean(param_chain):.5f}, std={np.std(param_chain):.5f}, min={np.min(param_chain):.5f}, max={np.max(param_chain):.5f}")

    # Debug info: acceptance fraction and autocorr time
    acc_frac = sampler.acceptance_fraction
    print(f"Mean acceptance fraction: {np.mean(acc_frac):.3f}")
    print(f"Acceptance fraction per walker: {acc_frac}")
    try:
        tau = sampler.get_autocorr_time()
        print(f"Autocorrelation times: {tau}")
    except Exception as e:
        print(f"Could not estimate autocorrelation time: {e}")

    return sampler

# =============================================================================
# 5) Analyze Samples ---------------------------------------------------------
def analyze_samples(sampler):
    """
    Generate trace plots and corner plot of the posterior samples.
    Returns the flattened, thinned sample array.
    """
    print("\n[5] Analyzing MCMC chains...")
    chain = sampler.get_chain()
    print("Chain summary statistics by parameter:")
    ndim = chain.shape[-1]
    for i in range(ndim):
        print(f"  Param {i}: min={np.min(chain[:,:,i]):.5f}, max={np.max(chain[:,:,i]):.5f}, mean={np.mean(chain[:,:,i]):.5f}, std={np.std(chain[:,:,i]):.5f}")

    nsteps = sampler.chain.shape[1]
    flat_samples = sampler.get_chain(discard=nsteps//4,
                                     thin=100,
                                     flat=True)
    print(f"  - Flat samples shape: {flat_samples.shape}")
    # Parameter summary
    print("Posterior parameter summary (median ±1σ):")
    labels = [
        't0','per','rp','a','inc',
        'TESS_u1','TESS_u2','TASTE_u1','TASTE_u2',
        'poly0','poly1','poly2','jit_TESS','jit_TASTE'
    ]
    for i, lab in enumerate(labels):
        mcmc = np.percentile(flat_samples[:, i], [15.865, 50, 84.135])
        q = np.diff(mcmc)
        print(f"  {lab}: {mcmc[1]:.5f} -{q[0]:.5f} +{q[1]:.5f}")

    # Trace plots
    fig, axes = plt.subplots(len(labels), 1, figsize=(10, 2*len(labels)), sharex=True)
    for i, lab in enumerate(labels):
        axes[i].plot(sampler.get_chain()[:,:,i], 'k', alpha=0.3)
        axes[i].set_ylabel(lab, fontsize=10)
        axes[i].tick_params(labelsize=8)
    axes[-1].set_xlabel('Step', fontsize=12)
    plt.tight_layout()
    plt.show()

    # Corner plot
    fig = corner.corner(flat_samples, labels=labels,
                        show_titles=True, title_fmt='.3f', label_kwargs={'fontsize':10})
    plt.show()

    return flat_samples

# =============================================================================
# 6) Best-Fit Model Comparison and Additional Plots -------------------------
def plot_best_fit(theta_best, taste_bjd, differential_allref,
                  tess_bjd, tess_flux, tess_flux_err):
    """
    Plot the best-fit transit model over the TESS and TASTE data,
    and produce a phase-folded light curve for TESS.
    """
    print("\n[6] Generating best-fit plots...")
    # Unpack best-fit parameters
    params = batman.TransitParams()
    params.t0, params.per = theta_best[0], theta_best[1]
    params.rp, params.a, params.inc = theta_best[2], theta_best[3], theta_best[4]
    params.ecc, params.w = 0.0, 90.0
    params.limb_dark = 'quadratic'
    params.u = theta_best[5:7]

    # TESS model
    m_tess = batman.TransitModel(params, tess_bjd)
    model_tess = m_tess.light_curve(params)

    # Plot TESS data + model
    plt.figure(figsize=(8,4))
    plt.errorbar(tess_bjd, tess_flux, yerr=tess_flux_err,
                 fmt='.', ms=2, label='TESS data', alpha=0.6)
    plt.plot(tess_bjd, model_tess, 'r-', lw=1.5, label='Best-fit model')
    plt.xlabel('BJD_TDB', fontsize=12)
    plt.ylabel('Relative flux', fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

    # Phase-folded TESS
    phase = ((tess_bjd - params.t0 + 0.5*params.per) % params.per) - 0.5*params.per
    sorted_idx = np.argsort(phase)
    phase_sorted = phase[sorted_idx]
    flux_sorted = tess_flux[sorted_idx]

    plt.figure(figsize=(8,4))
    plt.scatter(phase, tess_flux, s=2, alpha=0.5)
    plt.plot(phase_sorted, model_tess[sorted_idx], 'r-', lw=1.5)
    plt.xlim(-0.2, 0.2)
    plt.xlabel('Phase [days from mid-transit]', fontsize=12)
    plt.ylabel('Relative flux', fontsize=12)
    plt.tight_layout()
    plt.show()

    # TASTE model + trend
    params.u = theta_best[7:9]
    m_taste = batman.TransitModel(params, taste_bjd)
    base_taste = m_taste.light_curve(params)
    median_bjd = np.median(taste_bjd)
    trend = theta_best[9] + theta_best[10]*(taste_bjd-median_bjd) + theta_best[11]*(taste_bjd-median_bjd)**2
    model_taste = base_taste * trend

    # Plot TASTE data + model
    plt.figure(figsize=(8,4))
    plt.scatter(taste_bjd, differential_allref, s=2, alpha=0.5, label='TASTE data')
    plt.plot(taste_bjd, model_taste, 'r-', lw=1.5, label='Best-fit model')
    plt.xlabel('BJD_TDB', fontsize=12)
    plt.ylabel('Relative flux', fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

# =============================================================================
# 7) Derived Physical Parameters -------------------------------------------
def derive_physical_params(flat_samples):
    """
    Compute planet radius in Jupiter radii using posterior samples
    and a distribution for the stellar radius.
    """
    print("\n[7] Computing derived physical parameters...")
    # Scaled planet radius samples
    Rp_Rs = flat_samples[:,2]
    n_samples = len(Rp_Rs)
    # Stellar radius distribution: 0.48 ± 0.04 R_sun
    r_star = np.random.normal(0.48, 0.04, size=n_samples)  # in R_sun
    # Convert to Jupiter radii: R_sun/R_jup ≈ 695700 km / 71492 km
    conv = 695700/71492
    Rp = Rp_Rs * r_star * conv

    mean_Rp = np.mean(Rp)
    std_Rp  = np.std(Rp)
    print(f"Planet radius: {mean_Rp:.3f} ± {std_Rp:.3f} R_jup")

# =============================================================================
# Main execution ------------------------------------------------------------
if __name__ == "__main__":
    freeze_support()

    # Load data
    (taste_bjd, differential_allref, differential_allref_error,
     tess_bjd, tess_flux, tess_flux_err,
     taste_dir, tess_dir) = load_data()

    # Initialize parameters and boundaries
    theta0 = init_theta()
    boundaries = init_boundaries(theta0)

    # Quick check of initial log-probability
    initial_lp = log_probability(theta0,
                                 taste_bjd, differential_allref, differential_allref_error,
                                 tess_bjd, tess_flux, tess_flux_err,
                                 boundaries)
    print(f"Initial log-probability: {initial_lp:.2f}")
    initial_ll = log_likelihood(theta0,
                                taste_bjd, differential_allref, differential_allref_error,
                                tess_bjd, tess_flux, tess_flux_err)
    initial_pr = log_prior(theta0)
    print(f"  -> log-likelihood: {initial_ll:.2f}")
    print(f"  -> log-prior:      {initial_pr:.2f}")

    # Run MCMC
    sampler = run_mcmc(theta0,
                       taste_bjd, differential_allref, differential_allref_error,
                       tess_bjd, tess_flux, tess_flux_err,
                       tess_dir, boundaries)

    # Analyze samples
    flat_samples = analyze_samples(sampler)

    # Best-fit comparison plots
    theta_best = np.median(flat_samples, axis=0)
    plot_best_fit(theta_best,
                  taste_bjd, differential_allref,
                  tess_bjd, tess_flux, tess_flux_err)

    # Compute derived parameters
    derive_physical_params(flat_samples)

    print("\nAll steps completed successfully.")

    # -------------------------------------------------------------------------
    # Optional: differential evolution for starting point (commented out)
    # from pyde.de import DiffEvol
    # de = DiffEvol(lambda th: log_probability(th, taste_bjd, differential_allref,
    #                                         differential_allref_error,
    #                                         tess_bjd, tess_flux, tess_flux_err,
    #                                         boundaries), boundaries.T, nwalkers=50, maximize=True)
    # de.optimize(10000)
    # starting_point = de.population
    # -------------------------------------------------------------------------
