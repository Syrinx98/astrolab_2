#!/usr/bin/env python3
"""
science_analysis.py

Data reduction and time conversion script for TASTE science frames.
Based on the lesson from lecture03_science_analysis.pdf.

Steps:
1. Read science image list and select a subset for testing.
2. Data reduction: gain multiplication, bias subtraction, flat-field correction.
3. Propagate errors for the corrected frames.
4. Save corrected frames and associated error frames.
5. Extract observing metadata (JD, exposure time, airmass) from FITS headers.
6. Convert observation times to BJD_TDB and compute light travel time corrections.

Note: File paths remain as specified; ensure the directory structure is unchanged.
"""

import os
import pickle
import numpy as np
import warnings
from astropy.io import fits
from astropy import coordinates as coord, units as u
from astropy.time import Time
from matplotlib import pyplot as plt

# =============================================================================
# 3.1. Read the data
# =============================================================================

taste_dir = 'TASTE_analysis/group05_QATAR-1_20230212'
science_list_path = os.path.join(taste_dir, 'science', 'science.list')
print("\n--- 3.1 Read science image list ---")
print(f"Loading science list from: {science_list_path}")

# Load all science filenames
science_list = np.genfromtxt(science_list_path, dtype='str')
print(f"Found {len(science_list)} science frames.")

# For testing, work on a small subset
science_test_list = science_list[:10]
print(f"Using first {len(science_test_list)} frames for pipeline testing.\n")

# =============================================================================
# 3.2. Data reduction steps
# =============================================================================

print("--- 3.2 Data reduction parameters and calibration frames ---")

# Load calibration products
median_bias = pickle.load(open(f"{taste_dir}/bias/median_bias.p", "rb"))
bias_std = 1.3          # [e-] bias uncertainty per pixel
gain = 1.91            # [e-/ADU]
readout_noise = 7.4     # [e-] readout noise per pixel

median_flat = pickle.load(open(f"{taste_dir}/flat/median_normalized_flat.p", "rb"))
median_flat_err = pickle.load(open(f"{taste_dir}/flat/median_normalized_flat_errors.p", "rb"))

print("Calibration frames loaded:")
print("  - Median bias frame (ADU)")
print("  - Median normalized flat-field and its error")

# Suppress warnings due to zero-values in flat-field
warnings.filterwarnings('ignore', category=RuntimeWarning)
np.seterr(divide='ignore', invalid='ignore')

# Function to apply reduction to one frame

def reduce_frame(filename):
    """
    Apply gain correction, bias subtraction and flat-field correction.
    Returns corrected data and propagated error arrays.
    """
    # Load raw science data and convert to electrons
    with fits.open(os.path.join(taste_dir, 'science', filename)) as hdul:
        raw_data = hdul[0].data * gain

    # Subtract bias
    debiased = raw_data - median_bias

    # Flat-field correction
    corrected = debiased / median_flat

    # Error propagation: readout noise, bias error and photon noise
    error_debiased = np.sqrt(readout_noise**2 + bias_std**2 + np.maximum(debiased, 0))
    error_corrected = corrected * np.sqrt((error_debiased / np.where(debiased != 0, debiased, 1))**2 +
                                          (median_flat_err / median_flat)**2)
    return corrected, error_corrected

# Test reduction on subset
for fname in science_test_list:
    corrected, err = reduce_frame(fname)
print("Data reduction pipeline test completed for subset.\n")

# =============================================================================
# 3.3. Save the images
# =============================================================================

print("--- 3.3 Saving corrected frames and error maps ---")

# Ensure output directory exists
output_dir = os.path.join(taste_dir, 'correct')
os.makedirs(output_dir, exist_ok=True)

for fname in science_test_list:
    corrected, err = reduce_frame(fname)
    base = fname[:-5]  # remove .fits
    out_data = os.path.join(output_dir, base + '_corr.p')
    out_err  = os.path.join(output_dir, base + '_corr_errors.p')

    # Save with pickle
    pickle.dump(corrected, open(out_data, 'wb'))
    pickle.dump(err,      open(out_err,  'wb'))
    print(f"Saved: {out_data} and {out_err}")

print("All corrected frames and errors saved.\n")

# =============================================================================
# 3.4. Extracting and saving useful information
# =============================================================================

print("--- 3.4 Extracting observing metadata ---")

n = len(science_test_list)
jd_array = np.zeros(n)
exptime_array = np.zeros(n)
airmass_array = np.zeros(n)

for i, fname in enumerate(science_test_list):
    with fits.open(os.path.join(taste_dir, 'science', fname)) as hdul:
        hdr = hdul[0].header
        jd_array[i]       = hdr['JD']
        exptime_array[i]  = hdr['EXPTIME']
        airmass_array[i]  = hdr['AIRMASS']
        if i == 0:
            print(f"Header comments for first frame:")
            print(f"  JD     : {hdr.comments['JD']}")
            print(f"  EXPTIME: {hdr.comments['EXPTIME']}")
            print(f"  AIRMASS: {hdr.comments['AIRMASS']}\n")

print(f"Extracted metadata for {n} frames.\n")

# =============================================================================
# 3.5. Conversion to BJD_TDB
# =============================================================================

print("--- 3.5 Converting times to BJD_TDB ---")
# Define target coordinates and observatory location
target = coord.SkyCoord("20:13:31.61", "+65:09:43.49", unit=(u.hourangle, u.deg), frame='icrs')
location = ('45.8472d', '11.569d')  # Asiago - Cima Ekar

# Compute mid-exposure JD in UTC
mid_jd = jd_array + exptime_array / 86400.0 / 2.0

# Create Time object and compute light travel time correction
tm = Time(mid_jd, format='jd', scale='utc', location=location)
ltt = tm.light_travel_time(target)
bjd_tdb = tm.tdb + ltt

print(f"Computed BJD_TDB for {n} observations.")

# =============================================================================
# 3.6. Light Travel Time variation plot
# =============================================================================

print("--- 3.6 Plotting light travel time variation over a year ---")

# Year-long span for demonstration
jd_plot = np.arange(2460000, 2460365.25, 0.10)
tm_plot = Time(jd_plot, format='jd', scale='utc', location=location)
ltt_plot = tm_plot.light_travel_time(target, ephemeris='jpl')

# Plot
plt.figure(figsize=(8, 5))
plt.plot(jd_plot, ltt_plot.to_value(u.min))
plt.title('Light Travel Time Correction to Solar System Barycenter')
plt.xlabel('Julian Date (JD)')
plt.ylabel('Light travel time (minutes)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Print summary statistics
avg_ltt = np.mean(ltt.to_value(u.min))
dt_sec  = np.mean((mid_jd - bjd_tdb.to_value('jd')) * 86400.0)
print(f"Average light travel time: {avg_ltt:.2f} minutes")
print(f"Average difference between JD_UTC and BJD_TDB: {dt_sec:.2f} seconds")
