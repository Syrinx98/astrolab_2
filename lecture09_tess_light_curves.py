"""
9 Extracting TESS Light Curves and Applying Quality Flags

In this code, we demonstrate how to:

1. Open a TESS Light Curve file (LCf) corresponding to a single sector.
2. Explore the content of the LCf and identify the main columns of interest:
   - TIME (BJD_TDB)
   - SAP_FLUX and SAP_FLUX_ERR (Simple Aperture Photometry)
   - PDCSAP_FLUX and PDCSAP_FLUX_ERR (SAP flux corrected for systematics)
   - QUALITY bitmask
3. Convert the time array from BTJD to BJD by adding back the BJDREFI and BJDREFF constants.
4. Plot the SAP and PDCSAP fluxes to understand their differences.
5. Identify and exclude bad data points:
   - Use np.isfinite to remove NaN or Inf values.
   - Use the QUALITY bitmask to exclude observations with known issues.
6. Apply a conservative approach first (exclude all data with any nonzero quality flag).
7. Optionally, use a selective approach by defining a custom bitmask of problematic flags.
8. Further manually exclude data if necessary.
9. Save the final selected data into a pickle file for later analysis.

We assume:
- The LC file is already in the working directory.
- You have the sector's LC and TPF files named similarly as in the Moodle text.
- Here we show the analysis for sector 44 only. You must repeat the process for each sector.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pickle

# =============================================================================
print("\n\n 6.1. Reading TESS Light Curve File for a single sector")
print("=============================================================================\n")

"""
Replace the file name with your actual LC file name if needed.
In this example, we use sector 44:
LC file: GJ3470_sector44_lc.fits
"""

sector44_lcf = 'GJ3470_sector44_lc.fits'
print("Light Curve file being used:", sector44_lcf)

# Let's inspect the structure of the LC file
fits.info(sector44_lcf)

# Open the LC file
lchdu = fits.open(sector44_lcf)
print("\nColumns in the LIGHTCURVE extension:")
print(lchdu[1].columns)

# =============================================================================
print("\n\n 6.2. Extract arrays of interest")
print("=============================================================================\n")

"""
From the TESS documentation and the LC file columns, we identify the main arrays:
- TIME: BJD - 2457000
- SAP_FLUX, SAP_FLUX_ERR
- PDCSAP_FLUX, PDCSAP_FLUX_ERR
- QUALITY bitmask

We also need to add back BJDREFI and BJDREFF to TIME to get the actual BJD_TDB.
"""

sap_flux = lchdu[1].data['SAP_FLUX']
sap_flux_error = lchdu[1].data['SAP_FLUX_ERR']
pdcsap_flux = lchdu[1].data['PDCSAP_FLUX']
pdcsap_flux_error = lchdu[1].data['PDCSAP_FLUX_ERR']
quality_bitmask = lchdu[1].data['QUALITY']

time_array = lchdu[1].data['TIME'] + lchdu[1].header['BJDREFI'] +  lchdu[1].header['BJDREFF']

print("Number of data points:")
print("TIME:", time_array.shape)
print("SAP_FLUX:", sap_flux.shape)
print("PDCSAP_FLUX:", pdcsap_flux.shape)

# =============================================================================
print("\n\n 6.3. Plot SAP and PDCSAP fluxes")
print("=============================================================================\n")

"""
We make a basic plot of SAP and PDCSAP flux vs time.
"""

plt.figure(figsize=(6,4))
plt.scatter(time_array, sap_flux, s=5, label='SAP')
plt.scatter(time_array, pdcsap_flux, s=5, label='PDCSAP')
plt.xlabel('BJD_TDB [d]')
plt.ylabel('e-/s')
plt.legend()
plt.title("SAP and PDCSAP flux comparison")
plt.show()

# Notice some PDCSAP points are NaN and don't appear on the plot.
print("\nSome elements of PDCSAP flux:", pdcsap_flux[10:20])

# =============================================================================
print("\n\n 6.4. Excluding bad values (NaN/Inf) and bad QUALITY flags")
print("=============================================================================\n")

"""
We first exclude non-finite values.
We use np.isfinite to find good data points in PDCSAP (and thus also for SAP).
"""

finite_selection = np.isfinite(pdcsap_flux)

"""
We now exclude data points with quality flags > 0 (conservative approach).
"""

conservative_selection = ~(quality_bitmask > 0) & finite_selection

"""
At this point, conservative_selection is True where data is good and finite.
We can plot again highlighting excluded points.
"""

plt.figure(figsize=(6,4))
plt.scatter(time_array[conservative_selection], sap_flux[conservative_selection],
            s=5, label='SAP - selected data')
plt.scatter(time_array, pdcsap_flux, s=5, label='PDCSAP')
plt.scatter(time_array[~conservative_selection], sap_flux[~conservative_selection],
            s=5, c='r', label='SAP - excluded data')
plt.errorbar(time_array[conservative_selection], sap_flux[conservative_selection],
             yerr=sap_flux_error[conservative_selection], fmt=' ', alpha=0.5,
             ecolor='k', zorder=-1)

plt.xlabel('BJD_TDB [d]')
plt.ylabel('e-/s')
plt.title("TESS Lightcurve - sector 44 (conservative selection)", fontsize=12)
plt.legend()
plt.show()

# =============================================================================
print("\n\n 6.5. Optional manual exclusion of data ranges")
print("=============================================================================\n")

"""
If we notice transits or interesting features at times where data is partially excluded or 
not good, we may manually exclude them as well.

For example, let's remove data before BJD_TDB > 2459500.6 as in the example.
"""

final_selection = conservative_selection & (time_array > 2459500.6)

# Plot again highlighting the manually excluded data
plt.figure(figsize=(6,4))
plt.scatter(time_array[conservative_selection], sap_flux[conservative_selection],
            s=5, label='SAP - selected data')
plt.scatter(time_array, pdcsap_flux, s=5, label='PDCSAP')

plt.scatter(time_array[~conservative_selection], sap_flux[~conservative_selection],
            s=5, c='r', label='SAP - excluded data')
plt.scatter(time_array[~final_selection & conservative_selection],
            sap_flux[~final_selection & conservative_selection],
            s=20, c='y', marker='x', label='SAP - manually excluded')
plt.errorbar(time_array[conservative_selection], sap_flux[conservative_selection],
             yerr=sap_flux_error[conservative_selection], fmt=' ', alpha=0.5,
             ecolor='k', zorder=-1)

plt.xlabel('BJD_TDB [d]')
plt.ylabel('e-/s')
plt.title("TESS Lightcurve for GJ3470 - sector 44 (final selection)", fontsize=12)

plt.xlim(2459500.15, 2459500.80)
plt.ylim(10100, 10300)
plt.legend()
plt.show()

# =============================================================================
print("\n\n 6.6. Saving the selected data")
print("=============================================================================\n")

"""
We save the final selected data into a pickle file for later analysis. 
Remember to repeat this process for each sector you want to analyze.
"""

sector44_dictionary = {
    'time': time_array[final_selection],
    'sap_flux': sap_flux[final_selection],
    'sap_flux_error': sap_flux_error[final_selection],
    'pdcsap_flux': pdcsap_flux[final_selection],
    'pdcsap_flux_error': pdcsap_flux_error[final_selection]
}

pickle.dump(sector44_dictionary, open('GJ3470_TESS_sector044_selected.p','wb'))

print("Saved the final selected data for sector 44 in GJ3470_TESS_sector044_selected.p")
print("All steps completed.")