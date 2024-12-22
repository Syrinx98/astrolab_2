import numpy as np
import matplotlib.pyplot as plt
import pickle
import batman
from scipy import stats
import emcee
import corner
from multiprocessing import Pool

# =============================================================================
# Data Loading (Same as before)
# =============================================================================

print("\n\nLoading Data")
print("========================================\n")

taste_dir = "TASTE_analysis/group05_QATAR-1_20230212"
tess_dir = "TESS_analysis"

taste_bjd_tdb = pickle.load(open(f'{taste_dir}/taste_bjdtdb.p', 'rb'))
differential_allref = pickle.load(open(f'{taste_dir}/differential_allref.p', 'rb'))
differential_allref_error = pickle.load(open(f'{taste_dir}/differential_allref_error.p', 'rb'))

tess_sector_dict = pickle.load(open(f'{tess_dir}/qatar1_TESS_sector024_filtered.p', 'rb'))
tess_bjd_tdb = tess_sector_dict['time']
tess_normalized_flux = tess_sector_dict['selected_flux']
tess_normalized_ferr = tess_sector_dict['selected_flux_error']

print("Data loaded successfully.")

# =============================================================================
# Parameter Definition (Same as before)
# =============================================================================

print("\n\nDefining Parameters and Model Setup")
print("========================================\n")

theta_initial = np.empty(14)
theta_initial[0] = 2457475.204489  # t0 (mid-transit, BJD_TDB)
theta_initial[1] = 1.420024443     # per (days)
theta_initial[2] = 0.1463          # rp = Rp/Rs
theta_initial[3] = 6.25            # a = a/Rs
theta_initial[4] = 84.08           # inc (degrees)
theta_initial[5] = 0.51
theta_initial[6] = 0.10
theta_initial[7] = 0.65
theta_initial[8] = 0.08
theta_initial[9] = 0.245
theta_initial[10] = 0.0
theta_initial[11] = 0.0
theta_initial[12] = 0.0
theta_initial[13] = 0.0

# =============================================================================
# Log-Likelihood Function (Same as before)
# =============================================================================

print("\n\nDefining the log_likelihood function")
print("========================================\n")

def log_likelihood(theta):
    t0, per, rp, a, inc = theta[0:5]
    tess_u1, tess_u2 = theta[5], theta[6]
    taste_u1, taste_u2 = theta[7], theta[8]
    poly0, poly1, poly2 = theta[9], theta[10], theta[11]
    tess_jitter, taste_jitter = theta[12], theta[13]

    params = batman.TransitParams()
    params.t0 = t0
    params.per = per
    params.rp = rp
    params.a = a
    params.inc = inc
    params.ecc = 0.0
    params.w = 90.0
    params.limb_dark = "quadratic"

    params.u = [tess_u1, tess_u2]
    m_tess = batman.TransitModel(params, tess_bjd_tdb)
    tess_model_flux = m_tess.light_curve(params)

    params.u = [taste_u1, taste_u2]
    median_bjd = np.median(taste_bjd_tdb)
    poly_trend = poly0 + poly1 * (taste_bjd_tdb - median_bjd) + poly2 * (taste_bjd_tdb - median_bjd) ** 2
    m_taste = batman.TransitModel(params, taste_bjd_tdb)
    taste_model_flux = m_taste.light_curve(params) * poly_trend

    tess_errors_with_jitter = tess_normalized_ferr ** 2 + tess_jitter ** 2
    taste_errors_with_jitter = differential_allref_error ** 2 + taste_jitter ** 2

    chi2_tess = np.sum((tess_normalized_flux - tess_model_flux) ** 2 / tess_errors_with_jitter)
    chi2_taste = np.sum((differential_allref - taste_model_flux) ** 2 / taste_errors_with_jitter)

    N_tess = len(tess_normalized_flux)
    N_taste = len(differential_allref)
    N = N_tess + N_taste

    sum_ln_sigma_tess = np.sum(np.log(tess_errors_with_jitter))
    sum_ln_sigma_taste = np.sum(np.log(taste_errors_with_jitter))

    logL = -0.5 * (N * np.log(2.0 * np.pi) + chi2_tess + chi2_taste + sum_ln_sigma_tess + sum_ln_sigma_taste)
    return logL

print("log_likelihood at initial guess:", log_likelihood(theta_initial))

# =============================================================================
# Priors (Same as before)
# =============================================================================

print("\n\nDefining Priors")
print("========================================\n")

def log_prior(theta):
    tess_u1_mean, tess_u1_std = 0.51, 0.05
    tess_u2_mean, tess_u2_std = 0.10, 0.05
    taste_u1_mean, taste_u1_std = 0.65, 0.05
    taste_u2_mean, taste_u2_std = 0.08, 0.05

    lp = 0.0
    lp += np.log(stats.norm.pdf(theta[5], loc=tess_u1_mean, scale=tess_u1_std))
    lp += np.log(stats.norm.pdf(theta[6], loc=tess_u2_mean, scale=tess_u2_std))
    lp += np.log(stats.norm.pdf(theta[7], loc=taste_u1_mean, scale=taste_u1_std))
    lp += np.log(stats.norm.pdf(theta[8], loc=taste_u2_mean, scale=taste_u2_std))
    return lp

# =============================================================================
# Boundaries (Same as before)
# =============================================================================

print("\n\nDefining Boundaries and log_probability")
print("========================================\n")

boundaries = np.empty((2, 14))
boundaries[:, 0] = [theta_initial[0] - 0.5, theta_initial[0] + 0.5]
boundaries[:, 1] = [theta_initial[1] - 0.5, theta_initial[1] + 0.5]
boundaries[:, 2] = [0.0, 0.5]
boundaries[:, 3] = [0.0, 20.0]
boundaries[:, 4] = [0.0, 90.0]
boundaries[:, 5] = [0.0, 1.0]
boundaries[:, 6] = [0.0, 1.0]
boundaries[:, 7] = [0.0, 1.0]
boundaries[:, 8] = [0.0, 1.0]
boundaries[:, 9] = [0.0, 1.0]
boundaries[:, 10] = [-1.0, 1.0]
boundaries[:, 11] = [-1.0, 1.0]
boundaries[:, 12] = [0.0, 0.05]
boundaries[:, 13] = [0.0, 0.05]

def log_probability(theta):
    if np.any(theta < boundaries[0, :]) or np.any(theta > boundaries[1, :]):
        return -np.inf
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

print("log_probability at initial guess:", log_probability(theta_initial))

# =============================================================================
# MCMC Setup (Same as before)
# =============================================================================

print("\n\nRunning the MCMC Sampler with emcee")
print("========================================\n")

def run_mcmc(theta_initial, boundaries, nwalkers=50, nsteps=20000, filename=f'{tess_dir}/emcee_sampler_first_run.p'):
    ndim = len(theta_initial)
    starting_point = theta_initial + 1e-5 * np.random.randn(nwalkers, ndim)

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
        sampler.run_mcmc(starting_point, nsteps, progress=True)

    with open(filename, 'wb') as f:
        pickle.dump(sampler, f)
    print(f"MCMC run completed and results saved to '{filename}'.")

if __name__ == "__main__":
    run_mcmc(theta_initial, boundaries)

    # =============================================================================
    # MCMC Analysis
    # =============================================================================

    print("\n\nAnalyzing the MCMC Results")
    print("========================================\n")

    with open(f'{tess_dir}/emcee_sampler_first_run.p', 'rb') as f:
        sampler = pickle.load(f)

    burn_in = 2500
    thinning = 100
    flat_samples = sampler.get_chain(discard=burn_in, thin=thinning, flat=True)
    print("Flat samples shape:", flat_samples.shape)

    param_names = [
        "t0", "per", "rp", "a", "inc",
        "TESS_u1", "TESS_u2", "TASTE_u1", "TASTE_u2",
        "poly0", "poly1", "poly2",
        "jitter_TESS", "jitter_TASTE"
    ]

    # =============================================================================
    # Trace Plots
    # =============================================================================

    print("\n\nPlotting Trace Plots")
    print("========================================\n")

    fig, axes = plt.subplots(len(param_names), figsize=(10, 2 * len(param_names)), sharex=True)
    for i in range(len(param_names)):
        ax = axes[i]
        ax.plot(sampler.get_chain(discard=burn_in, thin=thinning)[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, sampler.get_chain(discard=burn_in, thin=thinning).shape[0])
        ax.set_ylabel(param_names[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("Step number")
    plt.suptitle("Trace Plots of Model Parameters")
    plt.show()

    # =============================================================================
    # Corner Plot
    # =============================================================================

    print("\n\nPlotting Corner Plot")
    print("========================================\n")

    fig = corner.corner(
        flat_samples,
        labels=param_names,
        show_titles=True,
        quantiles=[0.16, 0.5, 0.84],
        title_fmt=".5f",
        title_kwargs={"fontsize": 12}
    )
    fig.suptitle("Corner Plot of Posterior Distributions", fontsize=16)
    plt.show()

    # =============================================================================
    # Parameter Summary
    # =============================================================================

    print("\n\nParameter Summary")
    print("========================================\n")

    for i in range(len(theta_initial)):
        mcmc = np.percentile(flat_samples[:, i], [15.865, 50.0, 84.135])
        q = np.diff(mcmc)
        print(f"{param_names[i]}: {mcmc[1]:.7f} -{q[0]:.7f} +{q[1]:.7f}")

    # =============================================================================
    # Best-Fit Model and Plots
    # =============================================================================

    print("\n\nExtracting Best-Fit Model and Comparing with Data")
    print("========================================\n")

    theta_best = np.median(flat_samples, axis=0)
    print("Best-fit parameters (median of posterior):", theta_best)

    params_best = batman.TransitParams()
    params_best.t0 = theta_best[0]
    params_best.per = theta_best[1]
    params_best.rp = theta_best[2]
    params_best.a = theta_best[3]
    params_best.inc = theta_best[4]
    params_best.ecc = 0.0
    params_best.w = 90.0
    params_best.limb_dark = "quadratic"

    params_best.u = [theta_best[5], theta_best[6]]
    m_tess_best = batman.TransitModel(params_best, tess_bjd_tdb)
    tess_flux_model_best = m_tess_best.light_curve(params_best)

    params_best.u = [theta_best[7], theta_best[8]]
    median_bjd = np.median(taste_bjd_tdb)
    poly_trend_best = (
        theta_best[9]
        + theta_best[10] * (taste_bjd_tdb - median_bjd)
        + theta_best[11] * (taste_bjd_tdb - median_bjd) ** 2
    )
    m_taste_best = batman.TransitModel(params_best, taste_bjd_tdb)
    taste_flux_model_best = m_taste_best.light_curve(params_best) * poly_trend_best

    plt.figure(figsize=(10, 6))
    plt.scatter(taste_bjd_tdb, differential_allref, s=2, label='TASTE data', color='black')
    plt.plot(taste_bjd_tdb, taste_flux_model_best, lw=2, c='C1', label='Best-fit model')
    plt.xlabel("BJD TDB")
    plt.ylabel("Relative flux")
    plt.legend()
    plt.title("TASTE Data with Best-Fit Model")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(tess_bjd_tdb, tess_normalized_flux, s=2, label='TESS data', color='black')
    plt.plot(tess_bjd_tdb, tess_flux_model_best, lw=2, c='C1', label='Best-fit model')
    plt.xlabel("BJD TDB")
    plt.ylabel("Relative flux")
    plt.legend()
    plt.title("TESS Data with Best-Fit Model")
    plt.show()

    # =============================================================================
    # Folded Light Curve with Residuals
    # =============================================================================

    print("\n\nPlotting Folded Light Curve with Residuals")
    print("========================================\n")

    folded_tess_time_new = (tess_bjd_tdb - theta_best[0] - theta_best[1] / 2.) % theta_best[1] - theta_best[1] / 2.
    folded_range_new = np.arange(-theta_best[1] / 2., theta_best[1] / 2., 0.001)

    best_params_folded = batman.TransitParams()
    best_params_folded.t0 = 0.0
    best_params_folded.per = theta_best[1]
    best_params_folded.rp = theta_best[2]
    best_params_folded.a = theta_best[3]
    best_params_folded.inc = theta_best[4]
    best_params_folded.ecc = 0.0
    best_params_folded.w = 90.0
    best_params_folded.u = [theta_best[5], theta_best[6]]
    best_params_folded.limb_dark = "quadratic"

    best_model_folded = batman.TransitModel(best_params_folded, folded_range_new)
    best_tess_folded_flux = best_model_folded.light_curve(best_params_folded)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    plt.suptitle("Transit Light Curve of Qatar-1b (TESS Data, Best fit parameters)")

    axs[0].scatter(folded_tess_time_new, tess_normalized_flux, s=2, label='TESS folded data', color='black')
    axs[0].plot(folded_range_new, best_tess_folded_flux, lw=2, c='red', label='Folded Model')
    axs[0].set_xlim(-0.2, 0.2)
    axs[0].set_ylabel("Relative Flux")
    axs[0].legend()

    residuals = tess_normalized_flux - m_tess_best.light_curve(params_best)
    axs[1].scatter(folded_tess_time_new, residuals, s=2, color='black')
    axs[1].axhline(0, color='red', linestyle='--')
    axs[1].set_xlim(-0.2, 0.2)
    axs[1].set_xlabel("Time from mid-transit [days]")
    axs[1].set_ylabel("Residuals")

    plt.subplots_adjust(hspace=0)
    plt.show()

    # =============================================================================
    # Physical Parameter Derivation (Same as before)
    # =============================================================================

    print("\n\nDeriving Physical Parameters (Example: Planet Radius)")
    print("========================================\n")

    Rp_Rs_samples = flat_samples[:, 2]
    n_samples = len(Rp_Rs_samples)

    r_star_mean = 0.48
    r_star_std = 0.04
    r_star_samples = np.random.normal(r_star_mean, r_star_std, n_samples)

    R_sun_km = 695700.0
    R_jup_km = 71492.0
    factor_sun_to_jup = R_sun_km / R_jup_km

    Rp_jup = r_star_samples * Rp_Rs_samples * factor_sun_to_jup

    Rp_mean = np.mean(Rp_jup)
    Rp_std = np.std(Rp_jup)
    print(f"Planet radius [Rjup]: mean = {Rp_mean:.6f}, std = {Rp_std:.6f}")

    print("All steps completed successfully.")