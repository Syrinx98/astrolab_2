#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCMC Transit Fitting Script for Qatar-1b (ottimizzato, versione completa)

Modifiche principali:
 1) Pre-detrending del light curve TASTE con polinomio di grado 2 (fallback incluso)
 2) Caricamento dati TESS filtrati da Wotan (pdcsap_flat e pdcsap_err)
 3) Ricerca del punto di partenza con Differential Evolution
 4) Sampler emcee con StretchMove(a=1.3), 100 walker, 40000 passi
 5) Burn-in di 10000 passi e thinning di 50
 6) Diagnostic di acceptance fraction e autocorr time
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import batman
from scipy import stats
import emcee
from scipy.optimize import differential_evolution
from functools import partial
import warnings
import corner

# =============================================================================
# 1) Caricamento dati e pre-detrending TASTE --------------------------------
def load_data():
    taste_dir = 'TASTE_analysis/group05_QATAR-1_20230212'
    tess_dir  = 'TESS_analysis'

    # TASTE
    taste_bjd = pickle.load(open(f'{taste_dir}/taste_bjdtdb.p', 'rb'))
    flux      = pickle.load(open(f'{taste_dir}/differential_allref.p', 'rb'))
    ferr      = pickle.load(open(f'{taste_dir}/differential_allref_error.p', 'rb'))
    flux     /= np.median(flux)
    ferr     /= np.median(flux)

    # Pre-detrend: fit 2° polinomio su out-of-transit o fallback
    t0_guess  = 2457475.2045
    per_guess = 1.42002
    phase = ((taste_bjd - t0_guess + 0.5*per_guess) % per_guess) - 0.5*per_guess
    mask  = np.abs(phase) > 0.1
    if np.sum(mask) < 5:
        warnings.warn("Too few out-of-transit points; skipping pre-detrend.")
        flux_detr = flux.copy()
    else:
        coeff     = np.polyfit(taste_bjd[mask], flux[mask], deg=2)
        trend     = np.polyval(coeff, taste_bjd)
        flux_detr = flux / trend

    # TESS (dati filtrati con Wotan)
    tess_dict = pickle.load(open(f'{tess_dir}/qatar1_TESS_sector024_filtered.p', 'rb'))
    tess_bjd  = tess_dict['time']
    tess_flux = tess_dict['pdcsap_flat']
    tess_err  = tess_dict['pdcsap_err']
    tess_flux /= np.median(tess_flux)
    tess_err  /= np.median(tess_flux)

    return taste_bjd, flux_detr, ferr, tess_bjd, tess_flux, tess_err

# =============================================================================
# 2) Inizializzazione parametri e confini -----------------------------------
def init_theta():
    return np.array([
        2457475.2045,  # t0
        1.42002,       # per
        0.1463,        # rp/Rs
        6.25,          # a/Rs
        84.08,         # inc
        0.515, 0.098,  # TESS u1,u2
        0.652, 0.078,  # TASTE u1,u2
        1.0, 0.0, 0.0, # poly0, poly1, poly2
        0.001, 0.001   # jit_TESS, jit_TASTE
    ])

def init_boundaries(theta0):
    npar = len(theta0)
    bounds = np.zeros((2, npar))
    # t0, per
    bounds[:,0] = [theta0[0]-0.5, theta0[0]+0.5]
    bounds[:,1] = [theta0[1]-0.5, theta0[1]+0.5]
    # rp/Rs, a/Rs, inc
    bounds[:,2] = [0.0, 0.5]
    bounds[:,3] = [0.0, 20.0]
    bounds[:,4] = [0.0, 90.0]
    # limb darkening
    bounds[0,5:9] = 0.0
    bounds[1,5:9] = 1.0
    # trend coeffs
    bounds[:,9]  = [0.9,  1.1]
    bounds[:,10] = [-0.5, 0.5]
    bounds[:,11] = [-0.5, 0.5]
    # jitters
    bounds[:,12] = [0.0, 0.05]
    bounds[:,13] = [0.0, 0.05]
    return bounds

# =============================================================================
# 3) Funzioni di probabilità ------------------------------------------------
def log_prior(theta):
    u_means = [0.515, 0.098, 0.652, 0.078]
    u_stds  = [0.10,  0.10,  0.05,  0.10]
    lp = 0.0
    for i in range(4):
        lp += stats.norm.logpdf(theta[5+i], loc=u_means[i], scale=u_stds[i])
    return lp


def log_likelihood(theta, taste_bjd, flux, ferr, tess_bjd, tess_flux, tess_err):
    t0, per, rp, a, inc = theta[:5]
    tess_u1, tess_u2   = theta[5], theta[6]
    taste_u1, taste_u2 = theta[7], theta[8]
    poly0, poly1, poly2= theta[9:12]
    jit_tess, jit_taste= theta[12], theta[13]

    # batman setup
    params = batman.TransitParams()
    params.t0, params.per = t0, per
    params.rp, params.a   = rp, a
    params.inc            = inc
    params.ecc, params.w  = 0.0, 90.0
    params.limb_dark      = 'quadratic'

    # TESS model
    params.u = [tess_u1, tess_u2]
    m_tess  = batman.TransitModel(params, tess_bjd)
    model_tess = m_tess.light_curve(params)

    # TASTE model
    params.u = [taste_u1, taste_u2]
    m_taste = batman.TransitModel(params, taste_bjd)
    base_taste = m_taste.light_curve(params)
    trend = poly0 + poly1*(taste_bjd-np.mean(taste_bjd)) + poly2*(taste_bjd-np.mean(taste_bjd))**2
    model_taste = base_taste * trend

    # errori con jitter
    err2_tess  = tess_err**2  + jit_tess**2
    err2_taste = ferr**2      + jit_taste**2

    chi2_tess  = np.sum((tess_flux-model_tess)**2/err2_tess)
    chi2_taste = np.sum((flux-model_taste)**2/err2_taste)
    lnD = np.sum(np.log(err2_tess)) + np.sum(np.log(err2_taste))
    N   = len(tess_flux)+len(flux)
    return -0.5*(N*np.log(2*np.pi) + chi2_tess + chi2_taste + lnD)


def log_probability(theta, taste_bjd, flux, ferr, tess_bjd, tess_flux, tess_err, boundaries):
    lower, upper = boundaries
    if np.any(theta<lower) or np.any(theta>upper):
        return -np.inf
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, taste_bjd, flux, ferr, tess_bjd, tess_flux, tess_err)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

# =============================================================================
# 4) Differential Evolution per punto di partenza --------------------------
def find_starting_point(boundaries, taste_bjd, flux, ferr, tess_bjd, tess_flux, tess_err):
    bnds = [(lo,hi) for lo,hi in zip(boundaries[0], boundaries[1])]
    func = lambda x: -log_probability(x, taste_bjd, flux, ferr, tess_bjd, tess_flux, tess_err, boundaries)
    res = differential_evolution(func, bnds, maxiter=2000)
    print(f"DE start: log-prob = {-res.fun:.1f}")
    return res.x

# =============================================================================
# 5) Esecuzione MCMC --------------------------------------------------------
def run_mcmc(theta0, taste_bjd, flux, ferr, tess_bjd, tess_flux, tess_err, boundaries):
    ndim     = len(theta0)
    nwalkers = 100
    nsteps   = 40000
    p0 = theta0 + 1e-4 * np.random.randn(nwalkers, ndim)
    prob = partial(log_probability,
                   taste_bjd=taste_bjd, flux=flux, ferr=ferr,
                   tess_bjd=tess_bjd, tess_flux=tess_flux,
                   tess_err=tess_err, boundaries=boundaries)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, prob,
                                    moves=emcee.moves.StretchMove(a=1.3))
    print("Running MCMC...")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sampler.run_mcmc(p0, nsteps, progress=True)

    af = np.mean(sampler.acceptance_fraction)
    print(f"Mean acceptance fraction: {af:.3f}")
    try:
        tau = sampler.get_autocorr_time()
        print("Autocorr times:", np.round(tau,1))
    except Exception as e:
        print("Autocorr time estimate failed:", e)
    return sampler

# =============================================================================
# 6) Analisi catene: trace plot e corner -----------------------------------
def analyze_samples(sampler):
    labels = ['t0','per','rp','a','inc',
              'TESS_u1','TESS_u2','TASTE_u1','TASTE_u2',
              'poly0','poly1','poly2','jit_TESS','jit_TASTE']
    burn = 10000
    flat = sampler.get_chain(discard=burn, thin=50, flat=True)

    fig, axes = plt.subplots(len(labels),1, figsize=(8,2*len(labels)), sharex=True)
    for i, lab in enumerate(labels):
        axes[i].plot(sampler.get_chain()[:,:,i], alpha=0.3)
        axes[i].set_ylabel(lab)
    axes[-1].set_xlabel('Step')
    plt.tight_layout(); plt.show()

    fig = corner.corner(flat, labels=labels, show_titles=True, title_fmt='.3f')
    plt.show()
    return flat

# =============================================================================
# 7) Best-fit comparison plots ----------------------------------------------
def plot_best_fit(theta, taste_bjd, flux, ferr, tess_bjd, tess_flux, tess_err):
    params = batman.TransitParams()
    params.t0, params.per = theta[0], theta[1]
    params.rp, params.a, params.inc = theta[2], theta[3], theta[4]
    params.ecc, params.w = 0.0, 90.0
    params.limb_dark = 'quadratic'

    params.u = theta[5:7]
    m_tess = batman.TransitModel(params, tess_bjd)
    model_tess = m_tess.light_curve(params)

    plt.figure(figsize=(8,4))
    plt.errorbar(tess_bjd, tess_flux, yerr=tess_err, fmt='.', ms=2, alpha=0.5, label='TESS data')
    plt.plot(tess_bjd, model_tess, 'r-', lw=1.5, label='Best-fit model')
    plt.xlabel('BJD_TDB'); plt.ylabel('Relative flux'); plt.legend(); plt.tight_layout(); plt.show()

    phase = ((tess_bjd - params.t0 + 0.5*params.per) % params.per) - 0.5*params.per
    idx = np.argsort(phase)
    plt.figure(figsize=(8,4))
    plt.scatter(phase, tess_flux, s=2, alpha=0.5); plt.plot(phase[idx], model_tess[idx], 'r-', lw=1.5)
    plt.xlim(-0.2,0.2); plt.xlabel('Phase'); plt.ylabel('Relative flux'); plt.tight_layout(); plt.show()

    params.u = theta[7:9]
    m_taste = batman.TransitModel(params, taste_bjd)
    base_taste = m_taste.light_curve(params)
    trend = theta[9] + theta[10]*(taste_bjd-np.mean(taste_bjd)) + theta[11]*(taste_bjd-np.mean(taste_bjd))**2
    model_taste = base_taste * trend

    plt.figure(figsize=(8,4))
    plt.scatter(taste_bjd, flux, s=2, alpha=0.5, label='TASTE data')
    plt.plot(taste_bjd, model_taste, 'r-', lw=1.5, label='Best-fit model')
    plt.xlabel('BJD_TDB'); plt.ylabel('Relative flux'); plt.legend(); plt.tight_layout(); plt.show()

# =============================================================================
# 8) Derived physical parameters -------------------------------------------
def derive_physical_params(flat):
    RpRs = flat[:,2]
    n    = len(RpRs)
    r_star = np.random.normal(0.48, 0.04, size=n)
    conv = 695700/71492
    Rp = RpRs * r_star * conv
    print(f"Planet radius: {np.mean(Rp):.3f} ± {np.std(Rp):.3f} R_jup")

# =============================================================================
# Main ----------------------------------------------------------------------
if __name__ == '__main__':
    taste_bjd, flux_detr, ferr, tess_bjd, tess_flux, tess_err = load_data()
    theta0     = init_theta()
    boundaries = init_boundaries(theta0)
    theta0     = find_starting_point(boundaries, taste_bjd, flux_detr, ferr, tess_bjd, tess_flux, tess_err)
    sampler    = run_mcmc(theta0, taste_bjd, flux_detr, ferr, tess_bjd, tess_flux, tess_err, boundaries)
    flat       = analyze_samples(sampler)
    theta_med  = np.median(flat, axis=0)
    plot_best_fit(theta_med, taste_bjd, flux_detr, ferr, tess_bjd, tess_flux, tess_err)
    derive_physical_params(flat)

    print("Fitting completo.")