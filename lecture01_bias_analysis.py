# -*- coding: utf-8 -*-
"""
bias_analysis.py

Revision and enhancement of the bias analysis workflow from lecture01_bias_analysis.pdf
Features added/improved:
 - Comprehensive section headers and docstrings
 - Context managers to safely handle file I/O
 - Explicit closing of FITS files
 - More descriptive print statements and status updates
 - Proper plotting of both single and median bias images
 - Improved error handling and clarity on array shapes
 - Explanatory comments for each step
"""

import os
import glob
import pickle

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt

# -----------------------------
# 1.1 Dataset Download Reminder
# -----------------------------
print("\n\nSection 1.1: Download and move the dataset")
print("=" * 60)
print("Please download the dataset from:")
print("https://drive.google.com/drive/folders/17FBWwLrFVDm5L7hEC9ZELETUy-sMqhjF")
print("and place it in the TASTE_analysis folder under your home directory.")
print("Example:")
print("  mkdir -p ~/TASTE_analysis")
print("  mv ~/Downloads/group05_QATAR-1_20230212 ~/TASTE_analysis/")

# -----------------------------
# 1.2 Generate .list Files
# -----------------------------
print("\n\nSection 1.2: Writing FITS list files for bias, flat, and science")
print("=" * 60)
# Base data directory (adjust if needed)
taste_dir = 'TASTE_analysis/group05_QATAR-1_20230212'

# Subdirectories to index
subdirs = ['bias', 'flat', 'science']

for sub in subdirs:
    folder = os.path.join(taste_dir, sub)
    list_path = os.path.join(folder, f"{sub}.list")
    print(f"Scanning '{folder}' for .fits files...")
    fits_files = sorted(glob.glob(os.path.join(folder, '*.fits')))

    # Write the filenames (basename only) to the .list file
    with open(list_path, 'w') as lf:
        for fullpath in fits_files:
            lf.write(os.path.basename(fullpath) + '\n')
    print(f"Wrote {len(fits_files)} entries to {list_path}")

# Quick check: display the first few entries of bias.list
bias_list = np.genfromtxt(os.path.join(taste_dir, 'bias', 'bias.list'), dtype=str)
print("First 5 bias frames:", bias_list[:5])

# -----------------------------
# 1.3 Extract FITS Header Info
# -----------------------------
print("\n\nSection 1.3: Extracting header information from first bias frame")
print("=" * 60)
bias0_filename = bias_list[0]
bias0_path = os.path.join(taste_dir, 'bias', bias0_filename)

with fits.open(bias0_path) as hdulist:
    hdu = hdulist[0]
    header = hdu.header
    data_raw = hdu.data

print("Header keywords and values:")
for key in ['JD', 'AIRMASS', 'GAIN', 'RDNOISE', 'NAXIS1', 'NAXIS2']:
    comment = header.comments[key] if key in header.comments else ''
    print(f" - {key}: {header[key]}    # {comment}")

jd = header['JD']
gain = header['GAIN']
ron = header['RDNOISE']
naxis1 = header['NAXIS1']
naxis2 = header['NAXIS2']
print(f"Converted JD = {jd:.6f}, gain = {gain:.3f} e/ADU, readout noise = {ron:.3f} e")
# -----------------------------
# 1.4 Array Dimensions in Python
# -----------------------------
print("\n\nSection 1.4: Checking array shape vs. FITS NAXIS")
print("=" * 60)
# Convert to electrons
data_e = data_raw * gain
rows, cols = data_e.shape
print(f"FITS header NAXIS1 x NAXIS2: {naxis1} x {naxis2}")
print(f"NumPy array shape (rows x cols): {rows} x {cols}")
print(f"Array data type: {type(data_e)}")

# -----------------------------
# 1.5 Build 3D Data Stack
# -----------------------------
print("\n\nSection 1.5: Building 3D stack of bias frames")
print("=" * 60)
n_images = len(bias_list)
print(f"Number of bias frames: {n_images}")

# Prepare empty stack: shape = (n_images, rows, cols)
stack = np.empty((n_images, rows, cols), dtype=float)

# Fill the stack
for i, fname in enumerate(bias_list):
    path = os.path.join(taste_dir, 'bias', fname)
    with fits.open(path) as tmp:
        arr = tmp[0].data * gain
    stack[i, :, :] = arr
    if i < 3:
        print(f" Loaded {fname} into stack index {i}")
print("... completed loading all frames.")

# Compute median frame
dmedian = np.median(stack, axis=0)
print(f"Median bias frame shape: {dmedian.shape}")

# -----------------------------
# 1.6 Plot single vs. median
# -----------------------------
print("\n\nSection 1.6: Plotting a single frame and the median")
print("=" * 60)
fig, axs = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)

# Single frame
im1 = axs[0].imshow(data_e, origin='lower', vmin=data_e.min(), vmax=data_e.max())
axs[0].set_title('Single Bias Frame (raw)')
axs[0].set_xlabel('X [pixels]')
axs[0].set_ylabel('Y [pixels]')

# Median frame
im2 = axs[1].imshow(dmedian, origin='lower', vmin=dmedian.min(), vmax=dmedian.max())
axs[1].set_title('Median Bias Frame')
axs[1].set_xlabel('X [pixels]')
axs[1].set_ylabel('Y [pixels]')

# Shared colorbar
cbar = fig.colorbar(im1, ax=axs, fraction=0.046, pad=0.04)
cbar.set_label('Electrons [e]')
plt.show()

# -----------------------------
# 1.7 Statistical Analysis
# -----------------------------
print("\n\nSection 1.7: Statistical analysis of bias variations and noise")
print("=" * 60)

# Visual variation
fig, axs = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)
im_stat = axs[0].imshow(dmedian, origin='lower', vmin=np.percentile(dmedian, 5),
                       vmax=np.percentile(dmedian, 95))
axs[0].set_title('Median Bias with Tight Color Range')
axs[0].set_xlabel('X [pixels]')
axs[0].set_ylabel('Y [pixels]')

# Column-wise average
avg_cols = np.mean(dmedian, axis=0)
axs[1].plot(avg_cols)
axs[1].set_title('Column-wise Average of Median Bias')
axs[1].set_xlabel('Column index')
axs[1].set_ylabel('Mean counts [e]')
# Colorbar for top plot
cbar2 = fig.colorbar(im_stat, ax=axs[0], fraction=0.046, pad=0.04)
cbar2.set_label('Electrons [e]')
plt.show()

# Compute readout noise from header vs. data
start_col, end_col = 300, 350
std_single = np.std(data_e[:, start_col:end_col])
exp_noise_med = ron / np.sqrt(n_images)
exp_std_med = std_single / np.sqrt(n_images)
meas_std_med = np.std(dmedian[:, start_col:end_col])
pixel_errors = np.std(stack, axis=0) / np.sqrt(n_images)
med_pix_err = np.median(pixel_errors)

print(f"Header readout noise: {ron:.3f} e")
print(f"STD of single frame (cols {start_col}-{end_col}): {std_single:.3f} e")
print(f"Expected noise of median (fromRON/sqrt(N)): {exp_noise_med:.3f} e")
print(f"Expected STD of median (from data/sqrt(N)): {exp_std_med:.3f} e")
print(f"Measured STD of median (cols {start_col}-{end_col}): {meas_std_med:.3f} e")
print(f"Median pixel-based error: {med_pix_err:.3f} e")

# Distribution of pixel errors
t = pixel_errors.flatten()
plt.figure(figsize=(8, 6))
plt.hist(t, bins=20, range=(0, np.percentile(t, 99)), density=True,
         histtype='step', label='Pixel-based error')
plt.axvline(exp_noise_med, linestyle='--', label='RON/sqrt(N)')
plt.axvline(exp_std_med, linestyle='-.', label='STD/sqrt(N)')
plt.axvline(meas_std_med, linestyle=':', label='Measured STD of median')
plt.axvline(med_pix_err, linestyle='-', label='Median pixel error')
plt.xlabel('Error [e]')
plt.ylabel('Normalized density')
plt.legend()
plt.title('Error distribution of median bias')
plt.show()

# -----------------------------
# 1.8 Save results with pickle
# -----------------------------
print("\n\nSection 1.8: Saving outputs to pickle files")
print("=" * 60)

output_dir = os.path.join(taste_dir, 'bias')
pkl_paths = {
    'median_bias': dmedian,
    'median_error_map': pixel_errors,
    'median_error_value': med_pix_err,
    'bias_stack': stack
}

for name, obj in pkl_paths.items():
    file_path = os.path.join(output_dir, f"{name}.p")
    with open(file_path, 'wb') as pf:
        pickle.dump(obj, pf)
    print(f"Saved {name} to {file_path}")

# Load-check one of the pickles
loaded = pickle.load(open(os.path.join(output_dir, 'median_bias.p'), 'rb'))
print(f"Loaded median_bias with shape {loaded.shape}")
