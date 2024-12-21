"""
11 Creating a Transit Model with Batman

In this code, we demonstrate how to:

1. Load the TASTE and TESS light curves that we have selected as the best ones from previous analyses.
2. Use the batman package (Kreidberg 2015) to model a planetary transit given input parameters from literature (or ExoFOP).
3. Initially use approximate limb darkening coefficients and see that the model does not perfectly match the data.
4. Use ldtk (Parviainen & Aigrain 2015) to derive more accurate limb darkening coefficients for the star in the relevant filters.
5. Update the limb darkening coefficients in the model and show the improved agreement between model and data.

Note:
- Before running this code, ensure you have installed the packages batman-package, astroquery, ldtk==1.7, emcee, corner, pygtc.
- Adjust file names, filters, and parameters according to your target star and data.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pickle

# =============================================================================
print("\n\n 10.1. Load the Selected Light Curves")
print("=============================================================================\n")

tess_dir = "TESS_analysis"
taste_dir = "TASTE_analysis/group05_QATAR-1_20230212"
# Load TASTE data
taste_bjd_tdb = pickle.load(open(f'{taste_dir}/taste_bjdtdb.p', 'rb'))

differential_allref_normalized = pickle.load(open(f'{taste_dir}/differential_allref_normalized.p', 'rb'))
differential_allref_normalized_error = pickle.load(open(f'{taste_dir}/differential_allref_normalized_error.p', 'rb'))
differential_allref = pickle.load(open(f'{taste_dir}/differential_allref.p', 'rb'))
differential_allref_error = pickle.load(open(f'{taste_dir}/differential_allref_error.p', 'rb'))

# Load TESS data for a specific sector
tess_sector24_dict = pickle.load(open(f'{tess_dir}/qatar1_TESS_sector024_filtered.p', 'rb'))
tess_bjd_tdb = tess_sector24_dict['time']
tess_normalized_flux = tess_sector24_dict['selected_flux']
tess_normalized_ferr = tess_sector24_dict['selected_flux_error']

print("Data loaded successfully.")

# =============================================================================
print("\n\n 10.2. Install and test batman with a simple example")
print("=============================================================================\n")

# ATTENTION, to make it work on windows you have to install visual studio with c++ build tools (approx 10gb)
# in this way MSVC (cl.exe) compiler will be installed and batman will work
import batman

params = batman.TransitParams()
params.t0 = 0.0
params.per = 1.0
params.rp = 0.1
params.a = 15.0
params.inc = 87.0
params.ecc = 0.0
params.w = 90.0
params.u = [0.1, 0.3]
params.limb_dark = "quadratic"

t = np.linspace(-0.05, 0.05, 100)
m = batman.TransitModel(params, t)
flux = m.light_curve(params)

plt.figure(figsize=(6, 4))
plt.title("Test Batman Transit Model")
plt.plot(t, flux)
plt.xlabel("Time from mid-transit")
plt.ylabel("Relative flux")
plt.show()

# =============================================================================
print("\n\n 10.3. Set Realistic Parameters from Literature (Preliminary)")
print("=============================================================================\n")

# Example parameters (to be adjusted based on your target from ExoFOP)
params = batman.TransitParams()
params.t0 = 2457475.204489  # Tempo di congiunzione inferiore (BJD_TDB) :contentReference[oaicite:0]{index=0}
params.per = 1.420024443    # Periodo orbitale in giorni :contentReference[oaicite:1]{index=1}
params.rp = 0.1463          # Rapporto dei raggi planetario e stellare (Rp/Rs) :contentReference[oaicite:2]{index=2}
params.a = 6.25             # Semiasse maggiore in unità di raggi stellari (a/Rs) :contentReference[oaicite:3]{index=3}
params.inc = 84.08          # Inclinazione orbitale in gradi :contentReference[oaicite:4]{index=4}
params.ecc = 0.0            # Eccentricità orbitale :contentReference[oaicite:5]{index=5}
params.w = 90.0             # Argomento del periasse in gradi (non rilevante per orbite circolari)
params.u = [0.3, 0.1]       # Coefficienti di oscuramento al bordo (da determinare accuratamente)
params.limb_dark = "quadratic"  # Legge di oscuramento al bordo

# Plot model vs TASTE normalized data (just for visualization)
m_taste = batman.TransitModel(params, taste_bjd_tdb)
taste_model_flux = m_taste.light_curve(params)

plt.figure(figsize=(6, 4))
plt.title("TASTE Data vs Preliminary Model")
plt.scatter(taste_bjd_tdb, differential_allref_normalized, s=2)
plt.plot(taste_bjd_tdb, taste_model_flux, lw=2, c='C1', label='Prelim. model')
plt.xlabel("BJD_TDB")
plt.ylabel("Relative flux")
plt.legend()
plt.show()

# Plot model vs TESS data
m_tess = batman.TransitModel(params, tess_bjd_tdb)
tess_model_flux = m_tess.light_curve(params)

plt.figure(figsize=(6, 4))
plt.title("TESS Data vs Preliminary Model")
plt.scatter(tess_bjd_tdb, tess_normalized_flux, s=2)
plt.plot(tess_bjd_tdb, tess_model_flux, lw=2, c='C1', label='Prelim. model')
plt.xlabel("BJD_TDB")
plt.ylabel("Relative flux")
plt.legend()
plt.show()

# =============================================================================
print("\n\n 10.4. Deriving More Accurate Limb Darkening Coefficients with ldtk")
print("=============================================================================\n")

from ldtk import SVOFilter, LDPSetCreator

print("Available SVOFilter shortcuts:")
print(SVOFilter.shortcuts)

# Example filters: sloan_r and TESS
sloan_r = SVOFilter('SLOAN/SDSS.r')
tess_fr = SVOFilter('TESS')

# Plot filter profiles
fig, ax = plt.subplots(figsize=(6, 3))
sloan_r.plot(ax=ax)
tess_fr.plot(ax=ax)
ax.set_title("Filter Transmission Curves")
fig.tight_layout()
plt.show()

# Assume stellar parameters (teff, logg, z)
# These must be taken from literature or ExoFOP
teff = (4910 , 100)
logg = (4.55, 0.10)
z = (0.20, 0.10)

filters = [sloan_r, tess_fr]
sc = LDPSetCreator(teff=teff, logg=logg, z=z, filters=filters)
ps = sc.create_profiles(nsamples=2000)
ps.resample_linear_z(100)
ps.set_uncertainty_multiplier(10)
qm, qe = ps.coeffs_qd(do_mc=True, n_mc_samples=10000)
chains = np.array(ps._samples['qd'])

# Extract LD chains
u1_sloan_r_chains = chains[0, :, 0]
u2_sloan_r_chains = chains[0, :, 1]
u1_tess_chains = chains[1, :, 0]
u2_tess_chains = chains[1, :, 1]

print('Sloan r LD coefficients:',
      'u1 = {0:.2f} ± {1:.2f}'.format(np.mean(u1_sloan_r_chains), np.std(u1_sloan_r_chains)),
      'u2 = {0:.2f} ± {1:.2f}'.format(np.mean(u2_sloan_r_chains), np.std(u2_sloan_r_chains)))

print('TESS LD coefficients:',
      'u1 = {0:.2f} ± {1:.2f}'.format(np.mean(u1_tess_chains), np.std(u1_tess_chains)),
      'u2 = {0:.2f} ± {1:.2f}'.format(np.mean(u2_tess_chains), np.std(u2_tess_chains)))

# Choose final LD coefficients (with some rounding)
u1_sloan_r = 0.59
u2_sloan_r = 0.16
u1_tess = 0.35
u2_tess = 0.23

# =============================================================================
print("\n\n 10.5. Updating the Model with Improved Limb Darkening")
print("=============================================================================\n")

params = batman.TransitParams()
params.t0 = 2459500.53574
params.per = 3.3366510632883
params.rp = 0.0764
params.a = 13.94
params.inc = 88.9
params.ecc = 0.
params.w = 90.

# For TASTE:
params.u = [u1_sloan_r, u2_sloan_r]
params.limb_dark = "quadratic"

m_taste = batman.TransitModel(params, taste_bjd_tdb)
taste_model_flux = m_taste.light_curve(params)

plt.figure(figsize=(6, 4))
plt.title("TASTE Data vs Updated Model")
plt.scatter(taste_bjd_tdb, differential_allref_normalized, s=2)
plt.plot(taste_bjd_tdb, taste_model_flux, lw=2, c='C1', label='Updated LD')
plt.xlabel("BJD TDB")
plt.ylabel("Relative flux")
plt.legend()
plt.show()

# For TESS:
params.u = [u1_tess, u2_tess]
m_tess = batman.TransitModel(params, tess_bjd_tdb)
tess_model_flux = m_tess.light_curve(params)

plt.figure(figsize=(6, 4))
plt.title("TESS Data vs Updated Model")
plt.scatter(tess_bjd_tdb, tess_normalized_flux, s=2)
plt.plot(tess_bjd_tdb, tess_model_flux, lw=2, c='C1', label='Updated LD')
plt.xlabel("BJD TDB")
plt.ylabel("Relative flux")
plt.legend()
plt.show()

folded_tess_time = (tess_bjd_tdb - params.t0 - params.per / 2.) % params.per - params.per / 2.
folded_range = np.arange(- params.per / 2., params.per / 2., 0.001)
params.t0 = 0.0
m_folded_tess = batman.TransitModel(params, folded_range)
tess_folded_flux = m_folded_tess.light_curve(params)

plt.figure(figsize=(6, 4))
plt.title("TESS Folded Data vs Updated Model")
plt.scatter(folded_tess_time, tess_normalized_flux, s=2)
plt.plot(folded_range, tess_folded_flux, lw=2, c='C1', label='Updated LD')
plt.xlim(-0.2, 0.2)
plt.xlabel("Time from mid-transit [days]")
plt.ylabel("Relative flux")
plt.legend()
plt.show()

print("All steps completed. The updated model with new limb darkening coefficients shows better agreement with data.")
