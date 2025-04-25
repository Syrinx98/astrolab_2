import pickle
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt

# =============================================================================
# 2 Flat Analysis
# =============================================================================


# ----------------------------------------------------------------------------
# 2.1 Read the data
# ----------------------------------------------------------------------------
print("\n\n=== 2.1 Read the data ===")
# Directory containing the dataset (preserve as provided)
taste_dir = 'TASTE_analysis/group05_QATAR-1_20230212'

# Load list of flat frames
flat_list = np.genfromtxt(f'{taste_dir}/flat/flat.list', dtype=str)
print(f"Found {len(flat_list)} flat frames:\n{flat_list}")

# Load median bias and bias statistics from previous analysis
median_bias = pickle.load(open(f"{taste_dir}/bias/median_bias.p", "rb"))
bias_std = 1.3  # [e-] error on median bias
readout_noise = 7.4  # [e-] readout noise
gain = 1.91  # [e-/ADU]

# Open first flat frame to examine header and data
with fits.open(f'{taste_dir}/flat/{flat_list[0]}') as hdul:
    header = hdul[0].header
    flat00_data = hdul[0].data * gain

# Print CCD characteristics from header
print(f"CCD Gain         : {header['GAIN']:.2f} {header.comments['GAIN']}")
print(f"CCD Readout noise: {header['RDNOISE']:.2f} {header.comments['RDNOISE']}")
print(f"Image shape      : {header['NAXIS1']} x {header['NAXIS2']} pixels")

# ----------------------------------------------------------------------------
# 2.2 Overscan inspection
# ----------------------------------------------------------------------------
print("\n\n=== 2.2 Overscan inspection ===")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
im = ax1.imshow(flat00_data, origin='lower')
median_column = np.mean(flat00_data, axis=0)
ax2.plot(median_column)

# Add colorbar
cbar = fig.colorbar(im, ax=[ax1], fraction=0.046, pad=0.04)
cbar.set_label("Signal [e-]")

ax1.set_title('Flat Frame with Overscan Regions')
ax1.set_xlabel('X [pixels]')
ax1.set_ylabel('Y [pixels]')
ax2.set_title('Average Counts per Column')
ax2.set_xlabel('X [pixels]')
ax2.set_ylabel('Average Signal [e-]')
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------
# 2.3 Exclude overscan from statistics
# ----------------------------------------------------------------------------
print("\n\n=== 2.3 Exclude overscan ===")
# Determine vmin/vmax excluding 12-pixel-wide overscan on each side
vmin = np.min(flat00_data[:, 12:-12])
vmax = np.max(flat00_data[:, 12:-12])
print(f"Display range (excluding overscan): vmin = {vmin:.2f}, vmax = {vmax:.2f}")

fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(8, 6))
im2 = ax3.imshow(flat00_data, origin='lower', vmin=vmin, vmax=vmax)
median_column2 = np.mean(flat00_data, axis=0)
ax4.plot(median_column2)
ax4.set_ylim(vmin, vmax)

cbar2 = fig.colorbar(im2, ax=[ax3], fraction=0.046, pad=0.04)
cbar2.set_label("Signal [e-]")

ax3.set_title('Trimmed Display of Flat Frame')
ax3.set_xlabel('X [pixels]')
ax3.set_ylabel('Y [pixels]')
ax4.set_title('Average Counts per Column (trimmed)')
ax4.set_xlabel('X [pixels]')
ax4.set_ylabel('Average Signal [e-]')
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------
# 2.4 Compute normalization factors for each flat frame
# ----------------------------------------------------------------------------
print("\n\n=== 2.4 Compute normalization factors ===")
n_images = len(flat_list)
ny, nx = flat00_data.shape
stack = np.empty((n_images, ny, nx), dtype=float)

# Build stack of bias-subtracted, gain-corrected flats
for idx, fname in enumerate(flat_list):
    with fits.open(f'{taste_dir}/flat/{fname}') as hdul:
        data_e = hdul[0].data * hdul[0].header['GAIN']
    stack[idx] = data_e - median_bias
print(f"Stack dimensions: {stack.shape}")

# Define central box for reference median
win = 50
x0 = int(nx / 2 - win / 2)
x1 = int(nx / 2 + win / 2)
y0 = int(ny / 2 - win / 2)
y1 = int(ny / 2 + win / 2)
print(f"Central box coords: x=[{x0},{x1}), y=[{y0},{y1})")

# Calculate normalization factors and their uncertainties
normalization_factors = np.median(stack[:, y0:y1, x0:x1], axis=(1, 2))
norm_std = np.std(stack[:, y0:y1, x0:x1], axis=(1, 2)) / np.sqrt(win ** 2)
print(f"Normalization factors (per frame): {normalization_factors}")
print(f"Uncertainty on norms: {norm_std}")

# Plot normalization factors
plt.figure()
frames = np.arange(n_images)
plt.errorbar(frames, normalization_factors, yerr=norm_std, fmt='o', ms=4)
plt.title('Flat Field Normalization Factors')
plt.xlabel('Frame Index')
plt.ylabel('Median Signal in Central Box [e-]')
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------
# 2.5 Normalize the flat frames and compare methods
# ----------------------------------------------------------------------------
print("\n\n=== 2.5 Normalize flat frames ===")
# Method 1: iterative
stack_norm_iter = np.empty_like(stack)
for i in range(n_images):
    stack_norm_iter[i] = stack[i] / normalization_factors[i]

# Method 2: vectorized
stack_norm_vect = (stack.T / normalization_factors).T

diff = np.max(np.abs(stack_norm_iter - stack_norm_vect))
print(f"Max difference between normalization methods: {diff:.2e}")
print(f"Floating-point machine epsilon: {np.finfo(float).eps:e}")

# ----------------------------------------------------------------------------
# 2.6 Build and save median normalized flat and stacks
# ----------------------------------------------------------------------------
print("\n\n=== 2.6 Median normalized flat and save outputs ===")
median_flat = np.median(stack_norm_vect, axis=0)

# Save outputs
with open(f'{taste_dir}/flat/median_normalized_flat.p', 'wb', buffering=0) as f:
    pickle.dump(median_flat, f)
with open(f'{taste_dir}/flat/flat_normalized_stack.p', 'wb') as f:
    pickle.dump(stack_norm_vect, f)
with open(f'{taste_dir}/flat/flat_normalization_factors.p', 'wb') as f:
    pickle.dump(normalization_factors, f)
with open(f'{taste_dir}/flat/flat_stack.p', 'wb') as f:
    pickle.dump(stack, f)

# Display median flat
nmin = np.min(median_flat[:, 12:-12])
nmax = np.max(median_flat[:, 12:-12])

fig, (ax5, ax6) = plt.subplots(2, 1, figsize=(8, 6))
im3 = ax5.imshow(median_flat, origin='lower', vmin=nmin, vmax=nmax)
avg_col = np.mean(median_flat, axis=0)
ax6.plot(avg_col)
ax6.set_ylim(nmin, nmax)

cbar3 = fig.colorbar(im3, ax=[ax5], fraction=0.046, pad=0.04)
cbar3.set_label('Normalized Signal')

ax5.set_title('Median Normalized Flat')
ax5.set_xlabel('X [pixels]')
ax5.set_ylabel('Y [pixels]')
ax6.set_title('Average Column Value (median flat)')
ax6.set_xlabel('X [pixels]')
ax6.set_ylabel('Normalized Signal')
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------
# 2.7 Error propagation in combined flat
# ----------------------------------------------------------------------------
print("\n\n=== 2.7 Error propagation ===")
# Photon noise contribution: sqrt(signal)
photon_noise = np.sqrt(np.abs(stack))
total_error = np.sqrt(readout_noise ** 2 + bias_std ** 2 + photon_noise ** 2)
total_error_norm = (total_error.T / normalization_factors).T

# Combine errors by summing variances
median_flat_error = np.sum(total_error_norm ** 2, axis=0) / n_images
print(f"Median flat error array shape: {median_flat_error.shape}")

# Save error map
with open(f'{taste_dir}/flat/median_normalized_flat_errors.p', 'wb') as f:
    pickle.dump(median_flat_error, f)

# ----------------------------------------------------------------------------
# 2.8 Statistics on the flat before vs after normalization
# ----------------------------------------------------------------------------
print("\n\n=== 2.8 Statistics on flat ===")
# Define a small pixel box for comparison
sample_box = stack[:, 40:45, 250:255]
sample_norm = stack_norm_vect[:, 40:45, 250:255] * np.mean(normalization_factors)

plt.figure()
plt.hist(sample_box.flatten(), bins=20, alpha=0.5, label='Before normalization')
plt.hist(sample_norm.flatten(), bins=20, alpha=0.5, label='After normalization')
plt.title('Distribution of Counts Before and After Normalization')
plt.xlabel('Counts [e-]')
plt.ylabel('Number of Pixels')
plt.legend()
plt.tight_layout()
plt.show()

# Plot theoretical photon noise distribution for norms
mean_norm = np.mean(normalization_factors)
sigma_norm = np.sqrt(mean_norm)
x = np.linspace(np.min(normalization_factors), np.max(normalization_factors), 100)
y = 1 / (sigma_norm * np.sqrt(2 * np.pi)) * np.exp(- (x - mean_norm) ** 2 / (2 * sigma_norm ** 2))

plt.figure()
plt.hist(normalization_factors, bins=20, density=True, alpha=0.5, label='Measured norms')
plt.plot(x, y, label='Photon-noise model')
plt.title('Normalization Factor Distribution vs. Model')
plt.xlabel('Normalization Factor [e-]')
plt.ylabel('Probability Density')
plt.legend()
plt.tight_layout()
plt.show()
