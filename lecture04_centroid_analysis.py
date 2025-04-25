"""
Script for Centroid Measurement and Photometric Analysis
Based on Lecture 04: Centroid Measurement
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pickle
import time
from astropy.io import fits  # for FITS support if needed

# =============================================================================
# 1.1 Read and display the first corrected science frame
# =============================================================================
print("\n\n1.1 Reading the first corrected science frame...")
taste_dir = 'TASTE_analysis/group05_QATAR-1_20230212'
# Load list of science frames
science_list = np.genfromtxt(f'{taste_dir}/science/science.list', dtype=str)
print(f"Found {len(science_list)} science frames")

# Use first frame for testing
test_list = science_list[:10]
first_frame = test_list[0]
corr_file = f'{taste_dir}/correct/{first_frame[:-5]}_corr.p'
print(f"Loading corrected frame: {corr_file}")
science = pickle.load(open(corr_file, 'rb'))
print(f"Frame shape: {science.shape}")

# Estimate display limits on a central strip
vmin = np.min(science[:, 100:400])
vmax = np.max(science[:, 100:400])
print(f"Estimated vmin={vmin:.1f}, vmax={vmax:.1f}")
# Override vmax for better contrast
vmax_display = 5000

fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(
    science,
    origin='lower',
    cmap='magma',
    norm=LogNorm(vmin=vmin, vmax=vmax_display)
)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Flux [counts]')
ax.set_xlabel('X [pixels]')
ax.set_ylabel('Y [pixels]')
ax.set_title('Initial science frame (logarithmic scale)')
plt.tight_layout()
plt.show()

# Approximate star positions by eye
x_target, y_target = 416, 74
x_ref1, y_ref1 = 298, 107
x_ref2, y_ref2 = 117, 40
print(f"Approximate target coordinates: x={x_target}, y={y_target}")
print(f"Approximate reference star: x={x_ref1}, y={y_ref1}")
print(f"Approximate reference star: x={x_ref2}, y={y_ref2}")

# =============================================================================
# 1.2 Draw circles around target and reference for initial identification
# =============================================================================
from matplotlib.patches import Circle

def draw_circle(ax, x0, y0, radius, color, label=None):
    """Draws a circle centered at (x0, y0) with given radius."""
    circ = Circle((x0, y0), radius, edgecolor=color, facecolor='none', linewidth=1.5)
    ax.add_patch(circ)
    if label:
        ax.scatter([], [], c=color, label=label)

print("\n\n1.2 Plotting initial regions for star identification...")
fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(
    science,
    origin='lower',
    cmap='magma',
    norm=LogNorm(vmin=vmin, vmax=vmax_display)
)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Flux [counts]')
# Draw 9-pixel circles around the stars
draw_circle(ax, x_target, y_target, 9, 'white', 'Target (9 px)')
draw_circle(ax, x_ref1, y_ref1, 9, 'yellow', 'Reference star (9 px)')
draw_circle(ax, x_ref2, y_ref2, 9, 'cyan', 'Reference star (9 px)')
ax.set_xlabel('X [pixels]')
ax.set_ylabel('Y [pixels]')
ax.set_title('Initial star regions')
ax.legend(loc='upper right', framealpha=0.8)
plt.tight_layout()
plt.show()

# =============================================================================
# 1.3 Create annulus and 3D surface plots to inspect flux distribution
# =============================================================================
def draw_annulus(ax, x0, y0, inner_r, outer_r, color, alpha=0.5, label=None):
    """Draws an annulus between inner_r and outer_r at (x0, y0)."""
    theta = np.linspace(0, 2*np.pi, 200)
    radii = np.array([inner_r, outer_r])
    xs = np.outer(radii, np.cos(theta)) + x0
    ys = np.outer(radii, np.sin(theta)) + y0
    for i in range(len(radii)-1):
        ax.fill(xs[i:i+2], ys[i:i+2], facecolor=color, edgecolor='none', alpha=alpha)
    if label:
        ax.scatter([], [], c=color, label=label)

print("\n\n1.3 Drawing annulus around target and generating 3D plots...")
# 2D annulus plot
fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(
    science,
    origin='lower',
    cmap='magma',
    norm=LogNorm(vmin=vmin, vmax=vmax_display)
)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Flux [counts]')
draw_annulus(ax, x_target, y_target, 9, 15, 'white', alpha=0.5, label='Annulus 9–15 px')
ax.set_xlabel('X [pixels]')
ax.set_ylabel('Y [pixels]')
ax.set_title('Annulus for centroid computation')
ax.legend(loc='upper right', framealpha=0.8)
plt.tight_layout()
plt.show()

# 3D surface of full frame
print("\n3D flux distribution of the entire frame...")
y_len, x_len = science.shape
X, Y = np.meshgrid(np.arange(x_len), np.arange(y_len))
fig = plt.figure(figsize=(6, 4))
ax3d = fig.add_subplot(111, projection='3d')
surf = ax3d.plot_surface(
    X, Y, science,
    cmap='magma', norm=LogNorm(vmin=vmin, vmax=vmax_display),
    linewidth=0, antialiased=False
)
fig.colorbar(surf, ax=ax3d).set_label('Flux [counts]')
ax3d.set_title('3D flux distribution (full frame)')
ax3d.set_xlabel('X [pixels]')
ax3d.set_ylabel('Y [pixels]')
ax3d.set_zlabel('Flux')
plt.tight_layout()
plt.show()

# 3D zoom around target
print("\nZoomed 3D around target star...")
zoom_r = 15
vmax_zoom = 20000
sub = science[y_target-zoom_r:y_target+zoom_r, x_target-zoom_r:x_target+zoom_r]
Xs, Ys = np.meshgrid(
    np.arange(x_target-zoom_r, x_target+zoom_r),
    np.arange(y_target-zoom_r, y_target+zoom_r)
)
fig = plt.figure(figsize=(6, 4))
axz = fig.add_subplot(111, projection='3d')
surf2 = axz.plot_surface(
    Xs, Ys, sub,
    cmap='magma', norm=LogNorm(vmin=vmin, vmax=vmax_zoom),
    linewidth=0, antialiased=False
)
fig.colorbar(surf2, ax=axz).set_label('Flux [counts]')
axz.view_init(elev=30, azim=40)
axz.set_title('3D zoom on target')
axz.set_xlabel('X [pixels]')
axz.set_ylabel('Y [pixels]')
axz.set_zlabel('Flux')
plt.tight_layout()
plt.show()

# =============================================================================
# 1.5 Use meshgrid to accelerate distance computation and time performance
# =============================================================================
print("\n\n1.5 Comparing distance computation methods...")
# Print epoch time and elapsed
t0 = time.time()
print(f"Current time (s since epoch): {t0:.4f}")
t1 = time.time()
print(f"Time to print: {t1 - t0:.6f} s")

# Method 1: nested loops
t0 = time.perf_counter()
rr_loops = np.zeros_like(science, dtype=float)
for yi in range(y_len):
    for xi in range(x_len):
        rr_loops[yi, xi] = np.hypot(xi - x_target, yi - y_target)
elapsed_loops = time.perf_counter() - t0
print(f"Nested loops: {elapsed_loops:.4f} s")

# Method 2: vectorized meshgrid
t0 = time.perf_counter()
rr_mesh = np.hypot(X - x_target, Y - y_target)
elapsed_mesh = time.perf_counter() - t0
print(f"Meshgrid method: {elapsed_mesh:.4f} s")
print(f"Speedup: {elapsed_loops/elapsed_mesh:.1f}x")

# =============================================================================
# 1.6 Weighted centroid calculation and radius validation
# =============================================================================
print("\n\n1.6 Weighted centroid and radius check...")
inner_radius = 10
mask = rr_mesh < inner_radius
flux_vals = science[mask]
# Weighted sums
wx = (X[mask] * flux_vals).sum()
wy = (Y[mask] * flux_vals).sum()
total = flux_vals.sum()
x_cent = wx / total
y_cent = wy / total
print(f"Weighted centroid: x={x_cent:.2f}, y={y_cent:.2f}")

# Visualize good vs bad inner radius
print("Visual comparison of inner radii...")
fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(
    science, origin='lower', cmap='magma', norm=LogNorm(vmin=vmin, vmax=vmax_display)
)
cbar = fig.colorbar(im, ax=ax); cbar.set_label('Flux [counts]')
draw_circle(ax, x_cent, y_cent, inner_radius, 'white', 'Good radius (10 px)')
draw_circle(ax, x_cent, y_cent, 9, 'yellow', 'Bad radius (9 px)')
ax.set_xlim(x_cent-15, x_cent+15)
ax.set_ylim(y_cent-15, y_cent+15)
ax.set_title('Inner radius validation for centroid')
ax.legend(loc='upper right', framealpha=0.8)
plt.tight_layout()
plt.show()

# =============================================================================
# 1.7 Iterative centroid refinement
# =============================================================================
print("\n\n1.7 Iterative refinement of centroid...")
max_iter = 30
tol = 0.1  # percent change tolerance
x_prev, y_prev = x_cent, y_cent
for i in range(max_iter):
    mask = np.hypot(X - x_prev, Y - y_prev) < inner_radius
    flux_vals = science[mask]
    x_new = (X[mask] * flux_vals).sum() / flux_vals.sum()
    y_new = (Y[mask] * flux_vals).sum() / flux_vals.sum()
    dxp = (x_new - x_prev)/x_prev*100
    dyp = (y_new - y_prev)/y_prev*100
    print(f"Iter {i:2d}: x={x_new:.3f} ({dxp:.2f}%), y={y_new:.3f} ({dyp:.2f}%)")
    if abs(dxp) < 0.1 and abs(dyp) < 0.1:
        print("Convergence achieved")
        break
    x_prev, y_prev = x_new, y_new

# Convergence path plot
fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(
    science, origin='lower', cmap='magma', norm=LogNorm(vmin=vmin, vmax=vmax_display)
)
cbar = fig.colorbar(im, ax=ax); cbar.set_label('Flux [counts]')
draw_circle(ax, x_prev, y_prev, inner_radius, 'white', 'Final centroid (10 px)')
ax.plot([x_cent, x_prev], [y_cent, y_prev], 'o-', color='cyan', label='Centroid path')
ax.set_xlim(x_prev-15, x_prev+15)
ax.set_ylim(y_prev-15, y_prev+15)
ax.set_title('Centroid convergence')
ax.legend(loc='upper right', framealpha=0.8)
plt.tight_layout()
plt.show()

# --- 1.8 FWHM measurement via cumulative distribution, FIXED ---
print("\n\n1.8 FWHM measurement via cumulative distribution...")

# 1) Ensure no hidden NaNs/masked‐array issues:
if isinstance(science, np.ma.MaskedArray):
    img = science.filled(0)
else:
    img = np.nan_to_num(science, nan=0.0, posinf=0.0, neginf=0.0)

# 2) Build mask around your final centroid:
mask = np.hypot(X - x_prev, Y - y_prev) < inner_radius

# 3) Masked flux WITHOUT any invalid multiplies:
flux_masked = np.where(mask, img, 0.0)

# 4) Sum along each axis
flux_x = flux_masked.sum(axis=0)
flux_y = flux_masked.sum(axis=1)

# 5) Sanity check: make sure there's actually flux
total_x = flux_x.sum()
total_y = flux_y.sum()
if total_x <= 0 or total_y <= 0:
    raise RuntimeError(f"Zero (or negative) total flux! "
                       f"X‐sum={total_x}, Y‐sum={total_y}. "
                       "Check centroid or inner_radius.")

# 6) Normalized cumulative distributions
cum_x = np.cumsum(flux_x) / total_x
cum_y = np.cumsum(flux_y) / total_y

# 7) Distances from centroid
dist_x = np.arange(x_len) - x_prev
dist_y = np.arange(y_len) - y_prev

# 8) Plot if you like…
plt.figure(figsize=(6,4))
plt.plot(dist_x, cum_x, '.', label='NCD X-axis')
plt.plot(dist_y, cum_y, '.', label='NCD Y-axis')
for r in (+inner_radius, -inner_radius):
    plt.axvline(r, color='red', linestyle='--')
plt.axvline(0, color='k')
plt.xlabel('Distance from centroid [px]')
plt.ylabel('Normalized Cumulative Distribution')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# 9) Compute FWHM
def compute_fwhm(axis, ncd):
    left  = np.interp(0.15865, ncd, axis)
    right = np.interp(0.84135, ncd, axis)
    # right - left = 2·σ, FWHM = 2·√(2 ln 2)·σ
    return np.sqrt(2 * np.log(2)) * (right - left)

fwhm_x = compute_fwhm(dist_x, cum_x)
fwhm_y = compute_fwhm(dist_y, cum_y)
print(f"FWHM: X = {fwhm_x:.2f} px,  Y = {fwhm_y:.2f} px")

# 10) Finally convert to arcsec
bin_factor = 4
scale = 0.25  # arcsec per unbinned pixel
seeing_x = fwhm_x * bin_factor * scale
seeing_y = fwhm_y * bin_factor * scale
print(f"Estimated seeing: X = {seeing_x:.2f}\"  Y = {seeing_y:.2f}\"")
