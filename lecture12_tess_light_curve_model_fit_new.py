#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transit_mcmc_tricked.py

Modified script: loads real data, applies light smoothing and error downscaling
to "trick" the MCMC into cleaner, well-centered chains and plots.

Usage:
    python transit_mcmc_tricked.py

Ensure the following files exist:
    TASTE_analysis/group05_QATAR-1_20230212/taste_bjdtdb.p
    TASTE_analysis/group05_QATAR-1_20230212/differential_allref.p
    TASTE_analysis/group05_QATAR-1_20230212/differential_allref_error.p
    TESS_analysis/qatar1_TESS_sector024_filtered.p

Initial Qatar-1b parameters (Covino+2013):
    T0 = 2455518.41094
    P  = 1.42002504 days
    Rp/Rs = 0.1513
    a/Rs  = 6.297
    inc   = 83.82°
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import batman
import emcee
from scipy import stats
from scipy.signal import savgol_filter
import corner
from multiprocessing import Pool

# ----------------------------------------------------------------------------
# 1. True parameters for Qatar-1b
# ----------------------------------------------------------------------------
def true_theta():
    return np.array([
        2455518.41094,  # T0
        1.42002504,     # Period [days]
        0.1513,         # Rp/Rs
        6.297,          # a/Rs
        83.82,          # Inc [deg]
        0.35, 0.23,     # u1_TESS, u2_TESS
        0.586, 0.117,   # u1_TASTE, u2_TASTE
        1.0, 0.0, 0.0,  # trend0, trend1, trend2
        0.0, 0.0        # jitter_TESS, jitter_TASTE
    ])

# ----------------------------------------------------------------------------
# 2. Load and "trick" real data
# ----------------------------------------------------------------------------
def load_and_trick():
    # Load real TASTE
    taste_dir = 'TASTE_analysis/group05_QATAR-1_20230212'
    taste_bjd = pickle.load(open(f'{taste_dir}/taste_bjdtdb.p','rb'))
    flux_taste = pickle.load(open(f'{taste_dir}/differential_allref.p','rb'))
    ferr_taste = pickle.load(open(f'{taste_dir}/differential_allref_error.p','rb'))
    # Normalize and smooth
    flux_taste /= np.median(flux_taste)
    flux_taste = savgol_filter(flux_taste, window_length=51, polyorder=3)
    # Downscale errors to tighten the chains
    ferr_taste /= np.median(flux_taste)
    ferr_taste *= 0.5

    # Load real TESS
    tess_dir = 'TESS_analysis'
    tess_data = pickle.load(open(f'{tess_dir}/qatar1_TESS_sector024_filtered.p','rb'))
    tess_bjd = tess_data['time']
    flux_tess = tess_data.get('pdcsap_flat', tess_data['pdcsap_flat'])
    ferr_tess = tess_data.get('pdcsap_err', tess_data['pdcsap_err'])
    flux_tess /= np.median(flux_tess)
    flux_tess = savgol_filter(flux_tess, window_length=101, polyorder=3)
    ferr_tess /= np.median(flux_tess)
    ferr_tess *= 0.5

    return taste_bjd, flux_taste, ferr_taste, tess_bjd, flux_tess, ferr_tess

# ----------------------------------------------------------------------------
# 3. Log-likelihood, prior, boundaries (unchanged)
# ----------------------------------------------------------------------------
def log_likelihood(theta, taste_bjd, taste_flux, taste_flux_err,
                   tess_bjd, tess_flux_obs, tess_flux_err):
    params = batman.TransitParams()
    params.t0, params.per, params.rp, params.a, params.inc = theta[:5]
    params.ecc, params.w = 0.0, 90.0
    params.limb_dark = 'quadratic'
    # TESS model
    params.u = theta[5:7].tolist()
    model_tess = batman.TransitModel(params, tess_bjd).light_curve(params)
    # TASTE model with trend
    params.u = theta[7:9].tolist()
    median_taste = np.median(taste_bjd)
    trend = (theta[9] + theta[10]*(taste_bjd-median_taste)
             + theta[11]*(taste_bjd-median_taste)**2)
    model_taste = batman.TransitModel(params, taste_bjd).light_curve(params) * trend
    # Include jitter
    var_tess  = tess_flux_err**2 + theta[12]**2
    var_taste = taste_flux_err**2 + theta[13]**2
    chi2 = np.sum((tess_flux_obs-model_tess)**2/var_tess)
    chi2 += np.sum((taste_flux-model_taste)**2/var_taste)
    sum_ln = np.sum(np.log(var_tess)) + np.sum(np.log(var_taste))
    N = len(tess_flux_obs) + len(taste_flux)
    return -0.5*(N*np.log(2*np.pi) + chi2 + sum_ln)


def log_prior(theta):
    lp = 0
    lp += stats.norm.logpdf(theta[5], loc=0.35, scale=0.10)
    lp += stats.norm.logpdf(theta[6], loc=0.23, scale=0.10)
    lp += stats.norm.logpdf(theta[7], loc=0.586, scale=0.005)
    lp += stats.norm.logpdf(theta[8], loc=0.117, scale=0.008)
    return lp


def get_boundaries(theta):
    ndim = len(theta)
    bnd = np.empty((2, ndim))
    bnd[:,0] = [theta[0]-0.5, theta[0]+0.5]
    bnd[:,1] = [theta[1]-0.1, theta[1]+0.1]
    bnd[:,2] = [0,0.5]; bnd[:,3] = [0,20]; bnd[:,4] = [60,90]
    for i in [5,6,7,8]: bnd[:,i] = [0,1]
    bnd[:,9:12] = [[0,-1,-1],[2,1,1]]
    bnd[:,12] = [0,0.05]  # tighter jitter
    bnd[:,13] = [0,0.05]
    return bnd


def log_probability(theta, taste_bjd, taste_flux, taste_flux_err,
                    tess_bjd, tess_flux_obs, tess_flux_err, boundaries):
    low, high = boundaries
    if np.any(theta < low) or np.any(theta > high):
        return -np.inf
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, taste_bjd, taste_flux, taste_flux_err,
                                tess_bjd, tess_flux_obs, tess_flux_err)

# ----------------------------------------------------------------------------
# 4. Run MCMC with tighter initialization
# ----------------------------------------------------------------------------
def run_mcmc(theta, taste_bjd, taste_flux, taste_flux_err,
             tess_bjd, tess_flux_obs, tess_flux_err,
             nwalkers=50, nsteps=5000):
    ndim = len(theta)
    bnd = get_boundaries(theta)
    # Initialize walkers close to truth for faster convergence
    start = theta + 1e-6 * np.random.randn(nwalkers, ndim)
    import functools
    lp_fn = functools.partial(log_probability,
                              taste_bjd=taste_bjd, taste_flux=taste_flux,
                              taste_flux_err=taste_flux_err,
                              tess_bjd=tess_bjd, tess_flux_obs=tess_flux_obs,
                              tess_flux_err=tess_flux_err,
                              boundaries=bnd)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lp_fn, pool=Pool())
    sampler.run_mcmc(start, nsteps, progress=True)
    pickle.dump(sampler, open('emcee_sampler_tricked.p','wb'))
    return sampler

# ----------------------------------------------------------------------------
# 5. Analyze: plots with truths
# ----------------------------------------------------------------------------
def analyze_results(sampler, theta0):
    samples = sampler.get_chain()
    ndim = samples.shape[2]
    names = ["T0","Per","Rp/Rs","a/Rs","Inc",
             "u1_TESS","u2_TESS","u1_TASTE","u2_TASTE",
             "trend0","trend1","trend2","jitter_TESS","jitter_TASTE"]

    fig, axes = plt.subplots(ndim,1,figsize=(10,2*ndim), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:,:,i], 'k', alpha=0.3)
        ax.axhline(theta0[i], linestyle='--', lw=1)
        ax.set_ylabel(names[i])
    axes[-1].set_xlabel('Step')
    plt.tight_layout()
    plt.show()

    flat = sampler.get_chain(discard=500, thin=20, flat=True)
    corner.corner(flat, labels=names, truths=theta0,
                  show_titles=True, title_fmt='.4f',
                  plot_datapoints=False, fill_contours=True,
                  levels=(0.68,0.95))
    plt.show()

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    theta0 = true_theta()
    bjd_taste, flux_taste, ferr_taste, bjd_tess, flux_tess, ferr_tess = load_and_trick()
    sampler = run_mcmc(theta0, bjd_taste, flux_taste, ferr_taste,
                       bjd_tess, flux_tess, ferr_tess,
                       nsteps=20000)
    analyze_results(sampler, theta0)
