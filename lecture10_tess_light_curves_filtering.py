"""
10 Filtering TESS Light Curves using Wotan

In this code, we:

1. Import a previously selected TESS light curve (after quality checking and manual exclusions).
2. Use the wotan package to apply filtering algorithms (e.g., Huber spline 'hspline', 'biweight')
   to remove long-term trends and instrumental systematics from both SAP and PDCSAP flux.
3. Use a transit mask to exclude transit points from the filtering process, preventing the
   filtering algorithm from overfitting the transit signal.
4. Compare the standard deviation of the filtered light curves for different algorithms and
   window lengths to determine the best combination.
5. Save the resulting filtered light curve data for further analysis.

We assume:
- You have the TESS LC data already selected and saved in a pickle file, as done in the previous steps.
- The wotan package is installed (if not, run `pip install wotan`).
- The sklearn package is installed (if not, run `pip install scikit-learn`).

Remember to repeat a similar filtering process for each TESS sector you have.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from astropy.io import fits
from wotan import flatten, transit_mask

# =============================================================================
print("\n\n 8.1. Load previously selected TESS light curve data")
print("=============================================================================\n")

"""
We load the data from a dictionary created after quality checks and manual exclusions.
Replace 'qatar1_TESS_sector024_selected.p' with the appropriate file for your star and sector.
"""

tess_dir = "TESS_analysis"

sector23_file = f'{tess_dir}/qatar1_TESS_sector024_selected.p'
print("Loading data from:", sector23_file)

qatar1_TESS_sector024 = pickle.load(open(sector23_file, 'rb'))

time = qatar1_TESS_sector024['time']
sap_flux = qatar1_TESS_sector024['sap_flux']
sap_flux_error = qatar1_TESS_sector024['sap_flux_error']
pdcsap_flux = qatar1_TESS_sector024['pdcsap_flux']
pdcsap_flux_error = qatar1_TESS_sector024['pdcsap_flux_error']

print("Number of data points loaded:", len(time))

# =============================================================================
print("\n\n 8.2. Define transit parameters and transit mask")
print("=============================================================================\n")

"""
From ExoFOP or literature, we retrieve the Transit_time (T0), Period, and 
Transit_window (approx twice the transit duration in days).

We create a mask of in-transit points to exclude them from the flattening process.
"""

Transit_time = 2459688.443452
Period = 4.3011975
Transit_window = 2.982 * 2 /24.  # double the transit duration given in hours, convert to days

print("Transit_time:", Transit_time)
print("Period:", Period)
print("Transit_window (days):", Transit_window)

mask = transit_mask(
    time=time,
    period=Period,
    duration=Transit_window,
    T0=Transit_time
)
print("Number of in-transit points masked:", np.sum(mask))

# =============================================================================
print("\n\n 8.3. Test a simple filtering with Hspline (no mask first)")
print("=============================================================================\n")

sap_flatten_flux, sap_flatten_model = flatten(
    time,
    sap_flux,
    method='hspline',
    window_length=0.5,
    break_tolerance=0.5,
    return_trend=True
)

plt.figure(figsize=(8,4))
plt.title('TESS: original SAP LC and flattening model (no mask)')
plt.scatter(time, sap_flux, c='C0', s=3)
plt.errorbar(time, sap_flux, yerr=sap_flux_error, ecolor='k', fmt=' ', alpha=0.25, zorder=-1)
plt.plot(time, sap_flatten_model, c='C1', zorder=10, label='Hspline w:0.5 no mask')
plt.xlabel('BJD_TDB')
plt.ylabel('TESS SAP flux [e-/s]')
plt.legend()
plt.show()

# =============================================================================
print("\n\n 8.4. Apply masking to avoid including transit points in filtering")
print("=============================================================================\n")

sap_masked_flatten_flux, sap_masked_flatten_model = flatten(
    time,
    sap_flux,
    method='hspline',
    window_length=0.5,
    break_tolerance=0.5,
    return_trend=True,
    mask=mask
)

plt.figure(figsize=(8,4))
plt.title('TESS: SAP LC with and without mask')
plt.scatter(time, sap_flux, c='C0', s=2)
plt.errorbar(time, sap_flux, yerr=sap_flux_error, ecolor='k', fmt=' ', alpha=0.5, zorder=-1)
plt.plot(time, sap_flatten_model, c='C1', zorder=10, label='No mask')
plt.plot(time, sap_masked_flatten_model, c='C2', zorder=11, label='Mask')
plt.xlabel('BJD_TDB')
plt.ylabel('TESS SAP flux [e-/s]')
plt.legend()
plt.show()

# =============================================================================
print("\n\n 8.5. Phase-fold and check standard deviation around transit")
print("=============================================================================\n")

phase_folded_time = (time - Transit_time - Period/2) % Period - Period/2

print("STD with mask   :", np.std(sap_masked_flatten_flux[~mask]))
print("STD without mask:", np.std(sap_flatten_flux[~mask]))

# =============================================================================
print("\n\n 8.6. Test different algorithms and window lengths")
print("=============================================================================\n")

sap_masked_biweight_w10_flux, sap_masked_biweight_w10_model = flatten(
    time,
    sap_flux,
    method='biweight',
    window_length=1.0,
    break_tolerance=0.5,
    return_trend=True,
    mask=mask
)

sap_masked_biweight_w15_flux, sap_masked_biweight_w15_model = flatten(
    time,
    sap_flux,
    method='biweight',
    window_length=1.5,
    break_tolerance=0.5,
    return_trend=True,
    mask=mask
)

sap_masked_hspline_w10_flux, sap_masked_hspline_w10_model = flatten(
    time,
    sap_flux,
    method='hspline',
    window_length=1.0,
    break_tolerance=0.5,
    return_trend=True,
    mask=mask
)

sap_masked_hspline_w15_flux, sap_masked_hspline_w15_model = flatten(
    time,
    sap_flux,
    method='hspline',
    window_length=1.5,
    break_tolerance=0.5,
    return_trend=True,
    mask=mask
)

print('STD SAP biweight, window=1.0:', np.std(sap_masked_biweight_w10_flux[~mask]))
print('STD SAP biweight, window=1.5:', np.std(sap_masked_biweight_w15_flux[~mask]))
print('STD SAP hspline, window=1.0 :', np.std(sap_masked_hspline_w10_flux[~mask]))
print('STD SAP hspline, window=1.5 :', np.std(sap_masked_hspline_w15_flux[~mask]))

# Repeat for PDCSAP
pdcsap_masked_biweight_w10_flux, pdcsap_masked_biweight_w10_model = flatten(
    time,
    pdcsap_flux,
    method='biweight',
    window_length=1.0,
    break_tolerance=0.5,
    return_trend=True,
    mask=mask
)

pdcsap_masked_biweight_w15_flux, pdcsap_masked_biweight_w15_model = flatten(
    time,
    pdcsap_flux,
    method='biweight',
    window_length=1.5,
    break_tolerance=0.5,
    return_trend=True,
    mask=mask
)

pdcsap_masked_hspline_w05_flux, pdcsap_masked_hspline_w05_model = flatten(
    time,
    pdcsap_flux,
    method='hspline',
    window_length=0.5,
    break_tolerance=0.5,
    return_trend=True,
    mask=mask
)

pdcsap_masked_hspline_w10_flux, pdcsap_masked_hspline_w10_model = flatten(
    time,
    pdcsap_flux,
    method='hspline',
    window_length=1.0,
    break_tolerance=0.5,
    return_trend=True,
    mask=mask
)

pdcsap_masked_hspline_w15_flux, pdcsap_masked_hspline_w15_model = flatten(
    time,
    pdcsap_flux,
    method='hspline',
    window_length=1.5,
    break_tolerance=0.5,
    return_trend=True,
    mask=mask
)

print('STD PDCSAP biweight, w=1.0:', np.std(pdcsap_masked_biweight_w10_flux[~mask]))
print('STD PDCSAP biweight, w=1.5:', np.std(pdcsap_masked_biweight_w15_flux[~mask]))
print('STD PDCSAP hspline, w=0.5 :', np.std(pdcsap_masked_hspline_w05_flux[~mask]))
print('STD PDCSAP hspline, w=1.0 :', np.std(pdcsap_masked_hspline_w10_flux[~mask]))
print('STD PDCSAP hspline, w=1.5 :', np.std(pdcsap_masked_hspline_w15_flux[~mask]))

# Check average normalized error
avg_error = np.average(pdcsap_flux_error[~mask]/pdcsap_masked_biweight_w10_model[~mask])
print("average normalized error:", avg_error)

# =============================================================================
print("\n\n 8.7. Choose a final combination and save results")
print("=============================================================================\n")

"""
From the tests, we pick one combination that seems good. For example, 
pdcsap_masked_hspline_w10_flux and its associated error.

We will store several versions for flexibility.

WARNING: The code snippet in the Moodle text seems to reuse 'sap_masked_hspline_w10_flux' 
for pdcsap as well, probably a copy-paste error. Make sure to store the correct arrays.
We correct this by using the correct arrays: pdcsap_masked_hspline_w10_flux and so forth.
"""

sector23_dictionary = {
    'time': time,
    'selected_flux': pdcsap_masked_hspline_w10_flux,
    'selected_flux_error': pdcsap_flux_error/pdcsap_masked_hspline_w10_model,
    'sap_masked_hspline_w10_flux': sap_masked_hspline_w10_flux,
    'sap_masked_hspline_w10_flux_error': sap_flux_error/sap_masked_hspline_w10_model,
    'sap_masked_hspline_w15_flux': sap_masked_hspline_w15_flux,
    'sap_masked_hspline_w15_flux_error': sap_flux_error/sap_masked_hspline_w15_model,
    'pdcsap_masked_hspline_w10_flux': pdcsap_masked_hspline_w10_flux,
    'pdcsap_masked_hspline_w10_flux_error': pdcsap_flux_error/pdcsap_masked_hspline_w10_model,
    'pdcsap_masked_hspline_w15_flux': pdcsap_masked_hspline_w15_flux,
    'pdcsap_masked_hspline_w15_flux_error': pdcsap_flux_error/pdcsap_masked_hspline_w15_model,
}

output_filename = f'{tess_dir}/qatar1_TESS_sector024_filtered.p'
pickle.dump(sector23_dictionary, open(output_filename, 'wb'))
print("Filtered data saved in:", output_filename)
print("All steps completed.")
