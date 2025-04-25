#!/usr/bin/env python3
"""
4 Sky background subtraction and aperture photometry

Improved script with detailed comments, descriptive outputs, and enhanced error handling.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import pickle
from astropy.io import fits
import time
from scipy.stats import multivariate_normal

# =============================================================================
# 2.1. Read a science frame and its associated errors
# =============================================================================
print("\n\n 2.1 Reading science frame and error estimates")
print("="*80)

# Directory containing data
taste_dir = 'TASTE_analysis/group05_QATAR-1_20230212'

# Load list of science frames and select a subset for speed
science_list = np.genfromtxt(f'{taste_dir}/science/science.list', dtype=str)
science_temp_list = science_list[:10]

# Pick the 6th frame in the subset for analysis
frame_name = science_temp_list[5][:-5]
print(f"Loading corrected frame: {frame_name}_corr.p and its errors")

start_time = time.time()
try:
    science_corrected = pickle.load(open(f'{taste_dir}/correct/{frame_name}_corr.p', 'rb'))
    science_corrected_err = pickle.load(open(f'{taste_dir}/correct/{frame_name}_corr_errors.p', 'rb'))
    load_time = time.time() - start_time
    print(f"Loaded science frame and errors in {load_time:.2f} seconds")
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# =============================================================================
# 2.2. Compute distance arrays and identify target coordinates
# =============================================================================
print("\n\n 2.2 Computing coordinate grids and pixel distances")
print("="*80)

# Refined target position from previous analysis
x_target_refined = 415.851  # pixel
y_target_refined = 73.959   # pixel

# Frame dimensions
ylen, xlen = science_corrected.shape
X_axis = np.arange(xlen)
Y_axis = np.arange(ylen)
X, Y = np.meshgrid(X_axis, Y_axis)

# Distance of every pixel from the target center
target_distance = np.sqrt((X - x_target_refined)**2 + (Y - y_target_refined)**2)
print(f"Frame size: {xlen} x {ylen} pixels")
print(f"Target at (x, y) = ({x_target_refined:.3f}, {y_target_refined:.3f})")

# =============================================================================
# 2.3. Function to draw circles on the image
# =============================================================================
print("\n\n 2.3 Defining helper function to draw aperture and annulus")
print("="*80)

def make_circle_around_star(x_pos, y_pos, radius, thickness=0.5,
                            label='', color='white', alpha=1.0):
    """
    Draw a ring (annulus boundary) around a star at (x_pos, y_pos).

    Parameters:
    - x_pos, y_pos: center coordinates in pixels
    - radius: inner radius of ring in pixels
    - thickness: width of ring in pixels
    - label: legend label
    - color: edge color
    - alpha: transparency
    """
    # Create theta array for circle
    n = 100
    theta = np.linspace(0, 2*np.pi, n)
    # Radii for inner and outer edges of the ring
    radii = [radius, radius + thickness]
    # Coordinates for edges
    xs = np.outer(radii, np.cos(theta))
    ys = np.outer(radii, np.sin(theta))
    # Reverse second edge to form closed polygon
    xs[1, :] = xs[1, ::-1]
    ys[1, :] = ys[1, ::-1]
    # Plot the ring boundary
    ax = plt.gca()
    ax.fill(np.ravel(xs) + x_pos,
            np.ravel(ys) + y_pos,
            facecolor='none', edgecolor=color,
            linewidth=1.5, alpha=alpha, label=label)

# =============================================================================
# 2.4. Selecting the annulus for sky background measurement
# =============================================================================
print("\n\n 2.4 Visualizing annulus selection for sky background")
print("="*80)

inner_radius = 13  # pixels: must exclude target flux
outer_radius = 18  # pixels: include sufficient background

# Set display limits to highlight wings of PSF
vmin = np.amin(science_corrected[:, 100:400])
vmax = 2 * vmin
print(f"Display limits -> vmin: {vmin:.1f}, vmax: {vmax:.1f}")

fig, ax = plt.subplots(figsize=(6,6))
img = ax.imshow(science_corrected, cmap='magma',
                norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
                origin='lower', aspect='equal')
plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04, label='Counts')
# Zoom in around target
ax.set_xlim(x_target_refined - outer_radius*1.2,
            x_target_refined + outer_radius*1.2)
ax.set_ylim(y_target_refined - outer_radius*1.2,
            y_target_refined + outer_radius*1.2)
# Draw annulus boundaries
make_circle_around_star(x_target_refined, y_target_refined,
                        inner_radius, label='Inner radius', color='white')
make_circle_around_star(x_target_refined, y_target_refined,
                        outer_radius, label='Outer radius', color='yellow')
ax.set_xlabel('X [pixels]')
ax.set_ylabel('Y [pixels]')
ax.set_title('Annulus selection for sky background')
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()

# =============================================================================
# 2.5. Compute the sky background and its median
# =============================================================================
print("\n\n 2.5 Computing sky background level within annulus")
print("="*80)

annulus_selection = (target_distance > inner_radius) & (target_distance <= outer_radius)
n_annulus = np.sum(annulus_selection)

sky_flux_average = np.sum(science_corrected[annulus_selection]) / n_annulus
sky_flux_median = np.median(science_corrected[annulus_selection])

print(f"Pixels in annulus: {n_annulus}")
print(f"Average sky flux: {sky_flux_average:.2f} photons/pixel")
print(f"Median  sky flux: {sky_flux_median:.2f} photons/pixel")

# =============================================================================
# 2.5b. Compute associated error to the sky background
# =============================================================================
print("\n\n 2.5b Estimating uncertainty on sky background level")
print("="*80)

sky_flux_error = np.sqrt(np.sum(science_corrected_err[annulus_selection]**2)) / n_annulus
print(f"Sky background error: {sky_flux_error:.2f} photons/pixel")
# Consistency check
consistency = abs(sky_flux_average - sky_flux_median) <= sky_flux_error
print(f"Mean vs median consistent within error? {consistency}")

# Plot annulus selection overlay
plt.figure(figsize=(6,5))
plt.imshow(science_corrected, cmap='viridis', origin='lower',
           norm=plt.Normalize(vmin=vmin, vmax=vmax), aspect='equal')
plt.scatter(X[annulus_selection], Y[annulus_selection],
            s=1, color='red', alpha=0.6, label='Sky annulus')
plt.colorbar(label='Counts')
plt.title('Annulus used for sky measurement')
plt.xlabel('X [pixels]')
plt.ylabel('Y [pixels]')
plt.legend()
plt.tight_layout()
plt.show()

# =============================================================================
# 2.6. Remove the sky background from the measured flux of the star
# =============================================================================
print("\n\n 2.6 Subtracting sky background from entire frame")
print("="*80)

science_sky_corrected = science_corrected - sky_flux_average
science_sky_corrected_error = np.sqrt(science_corrected_err**2 + sky_flux_error**2)
print("Sky subtraction complete; arrays updated.")

# =============================================================================
# 2.7. Aperture photometry and error propagation
# =============================================================================
print("\n\n 2.7 Performing aperture photometry at multiple radii")
print("="*80)

radius_array = np.arange(0, inner_radius + 0.5, 0.5)
flux_vs_radius = np.zeros_like(radius_array)
error_vs_radius = np.zeros_like(radius_array)

for i, rad in enumerate(radius_array):
    sel = (target_distance <= rad)
    flux_vs_radius[i] = np.sum(science_sky_corrected[sel])
    error_vs_radius[i] = np.sqrt(np.sum(science_sky_corrected_error[sel]**2))

fractional_flux = flux_vs_radius / flux_vs_radius[-1]
fractional_error = error_vs_radius / flux_vs_radius[-1]

plt.figure(figsize=(7,4), dpi=300)
plt.errorbar(radius_array, fractional_flux, yerr=fractional_error,
             fmt='o', markersize=4, capsize=3, label='Measured')
for frac in [0.80, 0.85, 0.90, 0.95, 0.99]:
    plt.axhline(frac, linestyle='--', label=f'{int(frac*100)}% flux')
plt.xlabel('Aperture radius [pixels]')
plt.ylabel('Fractional enclosed flux')
plt.title('Encircled energy curve with measurement error')
plt.legend(loc='lower right', fontsize='small', ncol=2)
plt.tight_layout()
plt.show()

print("\nCandidate apertures:")
for r in [5, 13]:
    idx = np.argmin(abs(radius_array - r))
    print(f"  Radius = {radius_array[idx]:.1f} px -> Flux = {flux_vs_radius[idx]:.1f} ± {error_vs_radius[idx]:.1f}")

# =============================================================================
# 2.8. Comparing the star's profile with 2D Gaussian models
# =============================================================================
print("\n\n 2.8 Comparing measured profile to Gaussian PSF models")
print("="*80)

xy_range = np.arange(-inner_radius*1.2, inner_radius*1.2, 0.1)
Xg, Yg = np.meshgrid(xy_range, xy_range)
pos = np.dstack((Xg, Yg))
rg = np.sqrt(Xg**2 + Yg**2)
model_radii = np.arange(0, inner_radius, 0.1)

covariances = [5.0, 10.0, 20.0]
model_flux = {}
for cov in covariances:
    pdf = multivariate_normal(mean=[0, 0], cov=cov).pdf(pos)
    cum_flux = np.array([np.sum(pdf[rg <= rr]) for rr in model_radii])
    model_flux[cov] = cum_flux / cum_flux[-1]

plt.figure(figsize=(6,5))
plt.scatter(radius_array, fractional_flux, s=20, label='Measured', alpha=0.7)
for cov, mf in model_flux.items():
    plt.plot(model_radii, mf, label=f'Gaussian cov={cov}')

plt.xlabel('Aperture radius [pixels]')
plt.ylabel('Fractional enclosed flux')
plt.title('Data vs. Gaussian PSF models')
plt.legend()
plt.tight_layout()
plt.show()

print("\nAll computations complete.")
