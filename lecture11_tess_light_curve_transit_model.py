import numpy as np
import matplotlib.pyplot as plt
import pickle
import batman
from ldtk import SVOFilter, LDPSetCreator

# =============================================================================
# Example Code: Creating a Transit Model with Batman and ldtk
# =============================================================================
# This script demonstrates how to:
# 1. Load and visualize selected TASTE and TESS light curves for Qatar-1.
# 2. Use batman to model a planetary transit and refine limb darkening.
# 3. Compare initial and updated models against data.

# 1) Load the Selected Light Curves
# ----------------------------------
tess_dir  = "TESS_analysis"
taste_dir = "TASTE_analysis/group05_QATAR-1_20230212"

print("\n1) Loading selected light curves...")
# TASTE data (normalized differential flux)
taste_bjd       = pickle.load(open(f'{taste_dir}/taste_bjdtdb.p','rb'))
taste_flux      = pickle.load(open(f'{taste_dir}/differential_allref_normalized.p','rb'))
taste_flux_err  = pickle.load(open(f'{taste_dir}/differential_allref_normalized_error.p','rb'))
# TESS data (filtered & normalized)
tess_dict       = pickle.load(open(f'{tess_dir}/qatar1_TESS_sector024_filtered.p','rb'))
tess_bjd        = tess_dict['time']
# use pdcsap_flat / pdcsap_err keys
if 'pdcsap_flat' in tess_dict:
    tess_flux     = tess_dict['pdcsap_flat']
    tess_flux_err = tess_dict['pdcsap_err']
else:
    tess_flux     = tess_dict['selected_flux']
    tess_flux_err = tess_dict['selected_flux_error']
print("Data loaded successfully.")

# 2) Test batman installation with a toy model
# --------------------------------------------
print("\n2) Testing batman with a simple transit model...")
params_test = batman.TransitParams()
params_test.t0         = 0.0
params_test.per        = 1.0
params_test.rp         = 0.1
params_test.a          = 15.0
params_test.inc        = 87.0
params_test.ecc        = 0.0
params_test.w          = 90.0
params_test.u          = [0.1,0.3]
params_test.limb_dark  = "quadratic"

t_test  = np.linspace(-0.05,0.05,200)
mod_test= batman.TransitModel(params_test, t_test)
flux_test = mod_test.light_curve(params_test)

plt.figure(figsize=(6,4))
plt.title("Test Batman Transit Model")
plt.plot(t_test, flux_test, 'C1', label='Model')
plt.xlabel("Time from mid-transit [d]")
plt.ylabel("Relative flux")
plt.legend(); plt.tight_layout(); plt.show()

# 3) Preliminary literature parameters for Qatar-1b
# ------------------------------------------------
print("\n3) Loading preliminary Qatar-1b parameters...")
params = batman.TransitParams()
params.t0         = 2457475.204489
params.per        = 1.420024443
params.rp         = 0.1463
params.a          = 6.25
params.inc        = 84.08
params.ecc        = 0.0
params.w          = 90.0
params.u          = [0.3,0.1]
params.limb_dark  = "quadratic"
# Compute models
taste_model = batman.TransitModel(params, taste_bjd)
flux_taste  = taste_model.light_curve(params)
tess_model  = batman.TransitModel(params, tess_bjd)
flux_tess   = tess_model.light_curve(params)
# Plot
plt.figure(figsize=(6,4))
plt.title("TASTE vs Preliminary Model")
plt.errorbar(taste_bjd, taste_flux, yerr=taste_flux_err, fmt='.', ms=2, label='Data')
plt.plot(taste_bjd, flux_taste, 'C1-', label='Model')
plt.xlabel("BJD_TDB"); plt.ylabel("Relative flux"); plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(6,4))
plt.title("TESS vs Preliminary Model")
plt.errorbar(tess_bjd, tess_flux, yerr=tess_flux_err, fmt='.', ms=2, label='Data')
plt.plot(tess_bjd, flux_tess, 'C1-', label='Model')
plt.xlabel("BJD_TDB"); plt.ylabel("Relative flux"); plt.legend(); plt.tight_layout(); plt.show()

# 4) Derive improved limb darkening with ldtk
# ------------------------------------------
print("\n4) Deriving limb darkening coefficients...")
print("Available filters:", SVOFilter.shortcuts)
filter_r    = SVOFilter('SLOAN/SDSS.r')
filter_tess = SVOFilter('TESS')
# Stellar params
teff = (4910,100); logg=(4.55,0.1); feh=(0.20,0.1)
ld_creator = LDPSetCreator(teff=teff, logg=logg, z=feh, filters=[filter_r,filter_tess])
profiles   = ld_creator.create_profiles(nsamples=2000)
profiles.resample_linear_z(100); profiles.set_uncertainty_multiplier(10)
qm, qe    = profiles.coeffs_qd(do_mc=True, n_mc_samples=5000)
chains     = np.array(profiles._samples['qd'])
# Means
u1_r, u2_r  = np.mean(chains[0],axis=0)
u1_t, u2_t = np.mean(chains[1],axis=0)
print(f"SDSS r: u1={u1_r:.3f}, u2={u2_r:.3f}")
print(f"TESS   : u1={u1_t:.3f}, u2={u2_t:.3f}")

# 5) Update model with refined limb darkening
# -------------------------------------------
print("\n5) Updating transit model with refined limb darkening...")
for label,(bjd,flux,flux_err,u1,u2) in [
    ('TASTE',(taste_bjd, taste_flux, taste_flux_err, u1_r, u2_r)),
    ('TESS', (tess_bjd,  tess_flux,  tess_flux_err,  u1_t,  u2_t))]:
    params.u = [u1,u2]
    model = batman.TransitModel(params, bjd)
    flux_mod = model.light_curve(params)
    plt.figure(figsize=(6,4))
    plt.title(f"{label} vs Updated Model")
    plt.errorbar(bjd, flux, yerr=flux_err, fmt='.', ms=2, label='Data')
    plt.plot(bjd, flux_mod, 'C1-', label='Updated')
    plt.xlabel("BJD_TDB"); plt.ylabel("Relative flux"); plt.legend(); plt.tight_layout(); plt.show()

# 6) Phase-fold and compare for TESS
# ----------------------------------
folded_t = (tess_bjd - params.t0 - params.per/2) % params.per - params.per/2
phase = np.linspace(-params.per/2,params.per/2,1000)
# Build folded params manually
params_fold = batman.TransitParams()
params_fold.t0        = 0.0
params_fold.per       = params.per
params_fold.rp        = params.rp
params_fold.a         = params.a
params_fold.inc       = params.inc
params_fold.ecc       = params.ecc
params_fold.w         = params.w
params_fold.u         = params.u
params_fold.limb_dark = params.limb_dark
model_fold = batman.TransitModel(params_fold, phase)
flux_fold  = model_fold.light_curve(params_fold)

plt.figure(figsize=(6,4))
plt.title("TESS Folded Data vs Updated Model")
plt.scatter(folded_t, tess_flux, s=2, label='Folded Data')
plt.plot(phase, flux_fold, 'C1-', label='Model')
plt.xlim(-0.2,0.2)
plt.xlabel("Phase [d]"); plt.ylabel("Relative flux"); plt.legend(); plt.tight_layout(); plt.show()

print("All steps completed.")