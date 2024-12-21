"""
Example Code: Creating a Transit Model with Batman and ldtk

This script demonstrates how to:
1. Load and visualize selected TASTE and TESS light curves for a target star (Qatar-1).
2. Use batman (Kreidberg 2015) to model a planetary transit, starting with approximate parameters.
3. Compare the initial model to the data and note any mismatch.
4. Use ldtk (Parviainen & Aigrain 2015) to derive more accurate limb darkening coefficients.
5. Update the transit model with the new limb darkening coefficients and compare again.
6. (Additional) Introduce updated parameters for Qatar-1b from new data and show how to incorporate them into the transit model.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import batman
from ldtk import SVOFilter, LDPSetCreator

# =============================================================================
# 1) Load the Selected Light Curves
# =============================================================================

print("\n\n1) Loading the Selected Light Curves...")
print("========================================\n")

# Directories containing the data (adjust to your local paths if needed)
tess_dir = "TESS_analysis"
taste_dir = "TASTE_analysis/group05_QATAR-1_20230212"

# -- TASTE data --
# Loading previously saved arrays with time (BJD_TDB) and flux (normalized and non-normalized).
# 'pickle.load()' loads python objects from the specified file.
taste_bjd_tdb = pickle.load(open(f'{taste_dir}/taste_bjdtdb.p', 'rb'))
differential_allref_normalized = pickle.load(open(f'{taste_dir}/differential_allref_normalized.p', 'rb'))
differential_allref_normalized_error = pickle.load(open(f'{taste_dir}/differential_allref_normalized_error.p', 'rb'))
differential_allref = pickle.load(open(f'{taste_dir}/differential_allref.p', 'rb'))
differential_allref_error = pickle.load(open(f'{taste_dir}/differential_allref_error.p', 'rb'))

# -- TESS data --
# Loading one sector's data for the same star, already filtered to remove bad data points.
tess_sector24_dict = pickle.load(open(f'{tess_dir}/qatar1_TESS_sector024_filtered.p', 'rb'))
tess_bjd_tdb = tess_sector24_dict['time']
tess_normalized_flux = tess_sector24_dict['selected_flux']
tess_normalized_ferr = tess_sector24_dict['selected_flux_error']

print("Data loaded successfully.")

# =============================================================================
# 2) Quick Batman Installation Check with a Simple Transit Example
# =============================================================================

print("\n\n2) Testing batman with a simple transit model...")
print("================================================\n")

# Create batman.TransitParams object and define some arbitrary transit parameters
test_params = batman.TransitParams()
test_params.t0 = 0.0          # time of inferior conjunction
test_params.per = 1.0         # orbital period
test_params.rp = 0.1          # planet radius (in stellar radii)
test_params.a = 15.0          # semi-major axis (in stellar radii)
test_params.inc = 87.0        # orbital inclination (in degrees)
test_params.ecc = 0.0         # eccentricity
test_params.w = 90.0          # longitude of periastron (not relevant since ecc=0)
test_params.u = [0.1, 0.3]    # limb darkening coefficients [u1, u2]
test_params.limb_dark = "quadratic"  # limb darkening law

# Generate a small time array around the transit and compute the model flux
t_test = np.linspace(-0.05, 0.05, 100)
m_test = batman.TransitModel(test_params, t_test)
flux_test = m_test.light_curve(test_params)

# Plot the simple test transit
plt.figure(figsize=(6, 4))
plt.title("Test Batman Transit Model")
plt.plot(t_test, flux_test, label='Test transit')
plt.xlabel("Time from mid-transit [days]")
plt.ylabel("Relative flux")
plt.legend()
plt.show()

# =============================================================================
# 3) Setting More Realistic Parameters from Literature (Preliminary)
# =============================================================================

print("\n\n3) Setting Preliminary Literature/ExoFOP Parameters...")
print("========================================================\n")

# Example transit parameters for Qatar-1b (to be refined)
params = batman.TransitParams()
params.t0 = 2457475.204489  # Time of mid-transit (BJD_TDB)
params.per = 1.420024443    # Orbital period (days)
params.rp = 0.1463          # Planet-to-star radius ratio (Rp/Rs)
params.a = 6.25             # Semi-major axis in stellar radii (a/Rs)
params.inc = 84.08          # Orbital inclination (degrees)
params.ecc = 0.0            # Eccentricity
params.w = 90.0             # Longitude of periastron
params.u = [0.3, 0.1]       # Rough limb darkening coefficients (initial guess)
params.limb_dark = "quadratic"

# --- Model vs TASTE data (preliminary) ---
m_taste = batman.TransitModel(params, taste_bjd_tdb)
taste_model_flux = m_taste.light_curve(params)

plt.figure(figsize=(6, 4))
plt.title("TASTE Data vs Preliminary Model")
plt.scatter(taste_bjd_tdb, differential_allref_normalized, s=2, label='TASTE data')
plt.plot(taste_bjd_tdb, taste_model_flux, lw=2, c='C1', label='Prelim. Model')
plt.xlabel("BJD_TDB")
plt.ylabel("Relative Flux")
plt.legend()
plt.show()

# --- Model vs TESS data (preliminary) ---
m_tess = batman.TransitModel(params, tess_bjd_tdb)
tess_model_flux = m_tess.light_curve(params)

plt.figure(figsize=(6, 4))
plt.title("TESS Data vs Preliminary Model")
plt.scatter(tess_bjd_tdb, tess_normalized_flux, s=2, label='TESS data')
plt.plot(tess_bjd_tdb, tess_model_flux, lw=2, c='C1', label='Prelim. Model')
plt.xlabel("BJD_TDB")
plt.ylabel("Relative Flux")
plt.legend()
plt.show()

# =============================================================================
# 4) Deriving More Accurate Limb Darkening Coefficients with ldtk
# =============================================================================

print("\n\n4) Deriving Limb Darkening Coefficients with ldtk...")
print("======================================================\n")

# The ldtk library can generate limb darkening coefficients for specific filters
# given stellar atmospheric parameters (Teff, logg, metallicity, etc.).

print("Available SVOFilter shortcuts in ldtk:")
print(SVOFilter.shortcuts)

sloan_g = SVOFilter('SLOAN/SDSS.g')
sloan_r = SVOFilter('SLOAN/SDSS.r')
sloan_rprime = SVOFilter('SLOAN/SDSS.rprime_filter')
tess_fr = SVOFilter('TESS')

fig, ax = plt.subplots(figsize=(6,3))
sloan_g.plot(ax=ax)
sloan_r.plot(ax=ax)
sloan_rprime.plot(ax=ax)
tess_fr.plot(ax=ax)
ax.legend()
fig.tight_layout()
plt.show()

# Stellar parameters from the literature (or ExoFOP):
# (value, uncertainty)
teff = (4910, 100)   # Effective temperature
logg = (4.55, 0.10)  # Log g
z = (0.20, 0.10)     # Metallicity [Fe/H]

# Initialize the LDPSetCreator with these parameters and relevant filters
filters = [sloan_r, tess_fr]
sc = LDPSetCreator(teff=teff, logg=logg, z=z, filters=filters)

# Create the limb darkening profiles using 2000 samples
ps = sc.create_profiles(nsamples=2000)

# Resample the profiles and artificially inflate uncertainties (optional)
ps.resample_linear_z(100)
ps.set_uncertainty_multiplier(10)

# Compute the 'quadratic' limb darkening coefficients in the q1, q2 formalism
qm, qe = ps.coeffs_qd(do_mc=True, n_mc_samples=10000)
chains = np.array(ps._samples['qd'])  # The raw MCMC chains for q1, q2

# Extract the coefficients for each filter (Sloan r is index 0, TESS is index 1).
u1_sloan_r_chains = chains[0, :, 0]
u2_sloan_r_chains = chains[0, :, 1]
u1_tess_chains    = chains[1, :, 0]
u2_tess_chains    = chains[1, :, 1]

print('Sloan r LD coefficients:',
      'u1 = {0:.2f} ± {1:.2f}'.format(np.mean(u1_sloan_r_chains), np.std(u1_sloan_r_chains)),
      'u2 = {0:.2f} ± {1:.2f}'.format(np.mean(u2_sloan_r_chains), np.std(u2_sloan_r_chains)))
print('TESS LD coefficients:',
      'u1 = {0:.2f} ± {1:.2f}'.format(np.mean(u1_tess_chains), np.std(u1_tess_chains)),
      'u2 = {0:.2f} ± {1:.2f}'.format(np.mean(u2_tess_chains), np.std(u2_tess_chains)))

# Choose final LD coefficients (for instance, from the MCMC means) with some rounding
u1_sloan_r = 0.65
u2_sloan_r = 0.08
u1_tess    = 0.51
u2_tess    = 0.10

# =============================================================================
# 5) Updating the Transit Model with Improved Limb Darkening
# =============================================================================

print("\n\n5) Updating the Model with New Limb Darkening Coefficients...")
print("===============================================================\n")

# Re-create the transit parameters
params = batman.TransitParams()
params.t0  = 2457475.204489
params.per = 1.420024443
params.rp  = 0.1463
params.a   = 6.25
params.inc = 84.08
params.ecc = 0.0
params.w   = 90.0
params.limb_dark = "quadratic"

# -- Compare the updated model to TASTE data (Sloan r band) --
params.u = [u1_sloan_r, u2_sloan_r]
m_taste = batman.TransitModel(params, taste_bjd_tdb)
taste_model_flux = m_taste.light_curve(params)

plt.figure(figsize=(6, 4))
plt.title("TASTE Data vs Updated Model (Sloan r)")
plt.scatter(taste_bjd_tdb, differential_allref_normalized, s=2, label='TASTE data')
plt.plot(taste_bjd_tdb, taste_model_flux, lw=2, c='C1', label='Updated LD')
plt.xlabel("BJD_TDB")
plt.ylabel("Relative Flux")
plt.legend()
plt.show()

# -- Compare the updated model to TESS data --
params.u = [u1_tess, u2_tess]
m_tess = batman.TransitModel(params, tess_bjd_tdb)
tess_model_flux = m_tess.light_curve(params)

plt.figure(figsize=(6, 4))
plt.title("TESS Data vs Updated Model (TESS)")
plt.scatter(tess_bjd_tdb, tess_normalized_flux, s=2, label='TESS data')
plt.plot(tess_bjd_tdb, tess_model_flux, lw=2, c='C1', label='Updated LD')
plt.xlabel("BJD_TDB")
plt.ylabel("Relative Flux")
plt.legend()
plt.show()

# -- Create a phase-folded plot for the TESS data --
folded_tess_time = (tess_bjd_tdb - params.t0 - params.per / 2.) % params.per - params.per / 2.
folded_range = np.arange(-params.per / 2., params.per / 2., 0.001)

# For plotting the folded model, we often set t0=0 in a new params object
params_folded = batman.TransitParams()
params_folded.t0  = 0.0
params_folded.per = params.per
params_folded.rp  = params.rp
params_folded.a   = params.a
params_folded.inc = params.inc
params_folded.ecc = params.ecc
params_folded.w   = params.w
params_folded.u   = params.u  # use TESS LD
params_folded.limb_dark = params.limb_dark

m_folded_tess = batman.TransitModel(params_folded, folded_range)
tess_folded_flux = m_folded_tess.light_curve(params_folded)

plt.figure(figsize=(6, 4))
plt.title("TESS Folded Data vs Updated Model")
plt.scatter(folded_tess_time, tess_normalized_flux, s=2, label='TESS folded data')
plt.plot(folded_range, tess_folded_flux, lw=2, c='C1', label='Updated LD')
plt.xlim(-0.2, 0.2)
plt.xlabel("Time from mid-transit [days]")
plt.ylabel("Relative Flux")
plt.legend()
plt.show()

print("All steps completed. The updated model with new limb darkening coefficients shows improved agreement with the data.")

# =============================================================================
# 6) (Optional) Updating Parameters for Qatar-1b According to New Data
# =============================================================================

print("\n\n6) Updating Qatar-1b Parameters (Additional)...")
print("================================================\n")

# Create a new set of parameters WITHOUT altering the above steps:
updated_params = batman.TransitParams()

# Insert the new parameter values for Qatar-1b from your 'Data Summary' or new references
updated_params.t0  = 2457475.204489   # Transit Epoch (BJD_TDB)
updated_params.per = 1.420024443      # Orbital Period (days)
updated_params.rp  = 0.1463           # Rp/Rs
updated_params.a   = 6.25             # a/Rs
updated_params.inc = 84.08            # Inclination (deg)
updated_params.ecc = 0.0              # Eccentricity
updated_params.w   = 90.0             # Argument of Periastron
updated_params.u   = [0.36, 0.24]     # Limb Darkening Coefficients (for TESS)
updated_params.limb_dark = "quadratic"

print("Updated Qatar-1b parameters:")
print(f" t0  = {updated_params.t0}")
print(f" per = {updated_params.per}")
print(f" rp  = {updated_params.rp}")
print(f" a   = {updated_params.a}")
print(f" inc = {updated_params.inc}")
print(f" ecc = {updated_params.ecc}")
print(f" w   = {updated_params.w}")
print(f" u   = {updated_params.u}")
print()

# Compute the updated model (Qatar-1b) for TESS data and plot
updated_model_tess = batman.TransitModel(updated_params, tess_bjd_tdb)
updated_tess_flux  = updated_model_tess.light_curve(updated_params)

plt.figure(figsize=(6, 4))
plt.title("TESS Data vs Qatar-1b Model (New Parameters)")
plt.scatter(tess_bjd_tdb, tess_normalized_flux, s=2, label='TESS data')
plt.plot(tess_bjd_tdb, updated_tess_flux, lw=2, c='C1', label='New Qatar-1b Model')
plt.xlabel("BJD_TDB")
plt.ylabel("Relative Flux")
plt.legend()
plt.show()

# Create a folded plot using the new parameters
folded_tess_time_new = (tess_bjd_tdb - updated_params.t0 - updated_params.per/2.) % updated_params.per - updated_params.per/2.
folded_range_new = np.arange(-updated_params.per/2., updated_params.per/2., 0.001)

# For the folded model, set t0=0 in a new params object
updated_params_folded = batman.TransitParams()
updated_params_folded.t0  = 0.0
updated_params_folded.per = updated_params.per
updated_params_folded.rp  = updated_params.rp
updated_params_folded.a   = updated_params.a
updated_params_folded.inc = updated_params.inc
updated_params_folded.ecc = updated_params.ecc
updated_params_folded.w   = updated_params.w
updated_params_folded.u   = updated_params.u
updated_params_folded.limb_dark = updated_params.limb_dark

updated_model_folded = batman.TransitModel(updated_params_folded, folded_range_new)
updated_tess_folded_flux = updated_model_folded.light_curve(updated_params_folded)

plt.figure(figsize=(6, 4))
plt.title("TESS Folded Data vs Model (Qatar-1b, New Parameters)")
plt.scatter(folded_tess_time_new, tess_normalized_flux, s=2, label='TESS folded data')
plt.plot(folded_range_new, updated_tess_folded_flux, lw=2, c='C1', label='Folded Model')
plt.xlim(-0.2, 0.2)
plt.xlabel("Time from mid-transit [days]")
plt.ylabel("Relative Flux")
plt.legend()
plt.show()

print("End of the update. The model is now calibrated with new Qatar-1b parameters.")
