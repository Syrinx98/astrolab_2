"""
4 Centroid measurement
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle
from astropy.io import fits
import time
from matplotlib import colors

# =============================================================================
print("\n\n 1.1. Read the first science frame and identify the target and reference stars")
print("=============================================================================\n")

"""
Centroid measurement

We start by reading one image from the reduced and corrected science frames.
The frame used is the first of the test list.

We'll identify the target star and reference star positions approximately, 
then refine their positions.
"""

taste_dir = 'TASTE_analysis/group05_QATAR-1_20230212'

# Read the list of science frames
science_list = np.genfromtxt(f'{taste_dir}/science/science.list', dtype='str')
science_test_list = science_list[:10]
science_frame_name = f'{taste_dir}/correct/' + science_test_list[0][:-5] + '_corr.p'
science_corrected = pickle.load(open(science_frame_name, 'rb'))

# Estimate vmin and vmax for plotting
vmin = np.amin(science_corrected[:,100:400])
vmax = np.amax(science_corrected[:,100:400])
print('vmin:  {0:.1f}    vmax: {1:.1f}'.format(vmin, vmax))
vmax = 5000

fig, ax = plt.subplots(1, figsize=(8,3))
im1 = plt.imshow(science_corrected, cmap=plt.colormaps['magma'],
                 norm=colors.LogNorm(vmin=vmin, vmax=vmax), origin='lower')
plt.colorbar(im1,ax=ax, fraction=0.046, pad=0.04)
plt.xlabel(' X [pixels]')
plt.ylabel(' Y [pixels]')
plt.title("Initial frame for star identification")
plt.show()

# Approximate coordinates identified by inspecting the image
x_target = 416
y_target = 74

x_reference_01 = 298
y_reference_01 = 107

print("Approximate target coordinates: x={0}, y={1}".format(x_target, y_target))
print("Approximate reference star coordinates: x={0}, y={1}".format(x_reference_01, y_reference_01))

# =============================================================================
print("\n\n 1.2. Drawing circles around the identified stars")
print("=============================================================================\n")

"""
We use a helper function to draw circles around stars to refine the coordinates.
"""

def make_circle_around_star(x_pos, y_pos, label='', color='w'):
    from matplotlib.patches import Circle
    n, radii = 50, [9, 15]
    theta = np.linspace(0, 2*np.pi, n, endpoint=True)
    xs = np.outer(radii, np.cos(theta))
    ys = np.outer(radii, np.sin(theta))
    xs[1,:] = xs[1,::-1]
    ys[1,:] = ys[1,::-1]
    ax.fill(np.ravel(xs)+x_pos, np.ravel(ys)+y_pos, edgecolor=None, facecolor=color, alpha=0.75, label=label)


fig, ax = plt.subplots(1, figsize=(8,3))
im1 = plt.imshow(science_corrected, cmap=plt.colormaps['magma'],
                 norm=colors.LogNorm(vmin=vmin, vmax=vmax), origin='lower')
plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
make_circle_around_star(x_target, y_target, label='Target', color='w')
make_circle_around_star(x_reference_01, y_reference_01, label='Reference #1', color='y')
plt.xlabel(' X [pixels]')
plt.ylabel(' Y [pixels]')
ax.legend()
plt.title("Refining the approximate coordinates")
plt.show()

# =============================================================================
print("\n\n 1.3. Photocenter determination of the target star")
print("=============================================================================\n")

"""
We define another function to make a ring around the star and then plot again.
We will then proceed to do a 3D plot around the star to inspect its shape.
"""

def make_ring_around_star(x_pos, y_pos, label='', color='w'):
    from matplotlib.patches import Circle
    n, radii = 50, [9, 15]
    theta = np.linspace(0, 2*np.pi, n, endpoint=True)
    xs = np.outer(radii, np.cos(theta))
    ys = np.outer(radii, np.sin(theta))
    xs[1,:] = xs[1,::-1]
    ys[1,:] = ys[1,::-1]
    ax.fill(np.ravel(xs)+x_pos, np.ravel(ys)+y_pos, edgecolor=None, facecolor=color, alpha=0.75, label=label)

# Redoing the plot with rings
vmin = np.amin(science_corrected[:,100:400])
vmax = np.amax(science_corrected[:,100:400])
print('vmin:  {0:.1f}    vmax: {1:.1f}'.format(vmin, vmax))
vmax = 5000

fig, ax = plt.subplots(1, figsize=(8,3))
im1 = plt.imshow(science_corrected, cmap=plt.colormaps['magma'],
                 norm=colors.LogNorm(vmin=vmin, vmax=vmax), origin='lower')
plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
make_ring_around_star(x_target, y_target, label='Target', color='w')
make_ring_around_star(x_reference_01, y_reference_01, label='Reference #1', color='y')
plt.xlabel(' X [pixels]')
plt.ylabel(' Y [pixels]')
ax.legend()
plt.title("Target and Reference star with rings")
plt.show()

# =============================================================================
print("\n\n 1.4. Making a 3D plot of the target star region to understand flux distribution")
print("=============================================================================\n")

ylen, xlen  = np.shape(science_corrected)
print('Shape of our science frame: {0:d} x {1:d}'.format(xlen, ylen))
X_axis = np.arange(0, xlen, 1)
Y_axis = np.arange(0, ylen, 1)
X, Y = np.meshgrid(X_axis, Y_axis)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, science_corrected, cmap=plt.colormaps['magma'],
                       norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title("3D plot of the entire frame (not very informative)")
plt.show()

# A closer look around the target star
print("\nZooming around the target star with a more suitable vmax and radius...")
vmax= 20000
radius_plot = 15

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X[y_target-radius_plot:y_target+radius_plot, x_target-radius_plot:x_target+radius_plot],
                       Y[y_target-radius_plot:y_target+radius_plot, x_target-radius_plot:x_target+radius_plot],
                       science_corrected[y_target-radius_plot:y_target+radius_plot, x_target-radius_plot:x_target+radius_plot],
                       cmap=plt.colormaps['magma'], norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                       linewidth=0, antialiased=False)

ax.azim = 40 # viewing azimuth
ax.elev = 30 # viewing elevation
ax.set_xlabel('X [pixels]')
ax.set_ylabel('Y [pixels]')
ax.zaxis.set_rotate_label(False)
ax.set_zlabel('photoelectrons', rotation=90)

fig.colorbar(surf, shrink=0.5, aspect=15, ticks=[10, 100, 1000, 10000, 100000])
plt.title("3D zoom around target star")
plt.show()

# =============================================================================
print("\n\n 1.5. Using meshgrid to accelerate computations")
print("=============================================================================\n")

"""
We will measure the performance of two methods to compute the distance of each pixel from the star:
1) A double 'for' loop over x and y.
2) Using meshgrid arrays directly.

We use time.time() to measure execution times.
"""

t0 = time.time()
print('Seconds passed since January 1, 1970, 00:00:00 (UTC): {0:.4f}'.format(t0))
t1 = time.time()
delta_time = t1 - t0
print('Time spent to print the previous information: {0:f} seconds'.format(delta_time))

# First method (no meshgrid)
t0 = time.time()
rr_method01 = np.zeros_like(science_corrected)
for yi in range(0,np.shape(science_corrected)[0]):
    for xi in range (0, np.shape(science_corrected)[1]):
        rr_method01[yi, xi] =  np.sqrt((xi - x_target)**2 + (yi - y_target)**2)

t1 = time.time()
total_method01 = t1-t0
print('Time required by the first method: {0:f} seconds'.format(total_method01))

# Second method (using meshgrid)
t0 = time.time()
X_axis = np.arange(0, xlen, 1)
Y_axis = np.arange(0, ylen, 1)
X, Y = np.meshgrid(X_axis, Y_axis)
t1 = time.time()
rr_method02 = np.sqrt((X - x_target)**2 + (Y - y_target)**2)
t2 = time.time()

prepare_method02 = t1-t0
total_method02 = t2-t1
print('Time required to set up the second algorithm: {0:f} seconds'.format(prepare_method02))
print('Time required by second algorithm:            {0:f} seconds'.format(total_method02))
print('Algorithm using meshgrid is {0:.0f} times faster'.format(total_method01/total_method02))

# =============================================================================
print("\n\n 1.6. Centroid algorithm: Weighted centroid computation")
print("=============================================================================\n")

"""
We now define a weighted centroid algorithm. 
We pick an inner_radius that includes the star's flux and some background.
We compute a weighted average of pixel coordinates, weights are pixel fluxes.
"""

# Initial coordinates of the target (approximate)

inner_radius = 14

# We already have X, Y
target_distance = np.sqrt((X-x_target)**2 + (Y-y_target)**2)
annulus_sel = (target_distance < inner_radius)

weighted_X = np.sum(science_corrected[annulus_sel]*X[annulus_sel])
weighted_Y = np.sum(science_corrected[annulus_sel]*Y[annulus_sel])
total_flux = np.sum(science_corrected[annulus_sel])

x_target_refined = weighted_X/total_flux
y_target_refined = weighted_Y/total_flux

print('Initial coordinates  x: {0:5.2f}   y: {1:5.2f}'.format(x_target, y_target))
print('Refined coordinates  x: {0:5.2f}   y: {1:5.2f}'.format(x_target_refined, y_target_refined))

# Checking good vs bad inner radius visually
def make_circle_around_star(x_pos, y_pos, radius, thickness=0.5, label='', color='w', alpha=1.):
    from matplotlib.patches import Circle
    n, radii = 50, [radius, radius+thickness]
    theta = np.linspace(0, 2*np.pi, n, endpoint=True)
    xs = np.outer(radii, np.cos(theta))
    ys = np.outer(radii, np.sin(theta))
    xs[1,:] = xs[1,::-1]
    ys[1,:] = ys[1,::-1]
    ax.fill(np.ravel(xs)+x_pos, np.ravel(ys)+y_pos, edgecolor=None, facecolor=color, alpha=alpha, label=label)

vmin = np.amin(science_corrected[:,100:400])
vmax = np.amax(science_corrected[:,100:400])
print('vmin:  {0:.1f}    vmax: {1:.1f}'.format(vmin, vmax))
vmax = 2*vmin

fig, ax = plt.subplots(1, figsize=(5,5))
im1 = plt.imshow(science_corrected, cmap=plt.colormaps['magma'],
                 norm=colors.LogNorm(vmin=vmin, vmax=vmax), origin='lower')
plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
plt.xlim(x_target_refined-inner_radius*1.2, x_target_refined+inner_radius*1.2)
plt.ylim(y_target_refined-inner_radius*1.2, y_target_refined+inner_radius*1.2)

make_circle_around_star(x_target_refined, y_target_refined, inner_radius, label='Good inner radius')
make_circle_around_star(x_target_refined, y_target_refined, 9, color='y', label='Bad inner radius')

plt.xlabel(' X [pixels]')
plt.ylabel(' Y [pixels]')
plt.legend(loc='upper left')
plt.title("Inner radius visualization")
plt.show()

# =============================================================================
print("\n\n 1.7. Iterative centroid approach to check convergence")
print("=============================================================================\n")

"""
We run the centroid algorithm multiple times, using the refined coordinates as starting point 
for the next iteration, until convergence or maximum number of iterations is reached.
"""

x_target_initial = 416
y_target_initial = 74
maximum_number_of_iterations = 30

print('Initial coordinates  x: {0:5.2f}   y: {1:5.2f}'.format(x_target_initial, y_target_initial))

for i_iter in range(0, maximum_number_of_iterations):
    if i_iter == 0:
        x_target_previous = x_target_initial
        y_target_previous = y_target_initial
    else:
        x_target_previous = x_target_refined
        y_target_previous = y_target_refined

    target_distance = np.sqrt((X-x_target_previous)**2 + (Y-y_target_previous)**2)
    annulus_sel = (target_distance < inner_radius)

    weighted_X = np.sum(science_corrected[annulus_sel]*X[annulus_sel])
    weighted_Y = np.sum(science_corrected[annulus_sel]*Y[annulus_sel])
    total_flux = np.sum(science_corrected[annulus_sel])

    x_target_refined = weighted_X/total_flux
    y_target_refined = weighted_Y/total_flux

    percent_variance_x = (x_target_refined - x_target_previous)/x_target_previous * 100.
    percent_variance_y = (y_target_refined - y_target_previous)/y_target_previous * 100.

    print('    Iteration {0:3d}   x: {1:.3f} ({2:.2f}%)  y: {3:.3f} ({4:.2f}%)'.format(i_iter,
                                                                                     x_target_refined,
                                                                                     percent_variance_x,
                                                                                     y_target_refined,
                                                                                     percent_variance_y))
    if np.abs(percent_variance_x)<0.1 and  np.abs(percent_variance_y)<0.1:
        break

print('Refined coordinates  x: {0:5.2f}   y: {1:5.2f}'.format(x_target_refined, y_target_refined))

vmin = np.amin(science_corrected[:,100:400])
vmax = np.amax(science_corrected[:,100:400])
print('vmin:  {0:.1f}    vmax: {1:.1f}'.format(vmin, vmax))
vmax = 2*vmin

fig, ax = plt.subplots(1, figsize=(5,5))
im1 = plt.imshow(science_corrected, cmap=plt.colormaps['magma'],
                 norm=colors.LogNorm(vmin=vmin, vmax=vmax), origin='lower')
plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
plt.xlim(x_target_refined-inner_radius*1.2, x_target_refined+inner_radius*1.2)
plt.ylim(y_target_refined-inner_radius*1.2, y_target_refined+inner_radius*1.2)

make_circle_around_star(x_target_refined, y_target_refined, inner_radius, label='inner radius')
ax.scatter(x_target_initial, y_target_initial, c='C2', label='Starting point')
ax.scatter(x_target_refined, y_target_refined, c='k', label='Final point')

plt.xlabel(' X [pixels]')
plt.ylabel(' Y [pixels]')
plt.legend(loc='upper left')
plt.title("Convergence of the centroid position")
plt.show()

# =============================================================================
print("\n\n 1.8. Measuring the Full Width Half Maximum (FWHM)")
print("=============================================================================\n")

"""
We now measure how 'large' the star is. We compute the normalized cumulative distribution of flux 
along X and Y axes around the star and from that we determine the FWHM.

We assume no sky subtraction for now, just to show the method.
"""

target_distance = np.sqrt((X - x_target_refined)**2 + (Y - y_target_refined)**2)
annulus_sel = (target_distance < inner_radius)

# sums
total_flux = np.nansum(science_corrected*annulus_sel)
flux_x = np.nansum(science_corrected*annulus_sel, axis=0)
flux_y = np.nansum(science_corrected*annulus_sel, axis=1)

cumulative_sum_x = np.cumsum(flux_x)/total_flux
cumulative_sum_y = np.cumsum(flux_y)/total_flux

plt.figure()
plt.scatter(X_axis - x_target_refined, cumulative_sum_x, label='NCD along the X axis')
plt.scatter(Y_axis - y_target_refined, cumulative_sum_y, label='NCD along the Y axis')
plt.axvline(0, c='k')
plt.xlim(-inner_radius*1.3, inner_radius*1.3)
plt.axvline(inner_radius, c='C5', label='Inner radius')
plt.axvline(-inner_radius, c='C5')
plt.xlabel('Distance from the photocenter [pixels]')
plt.ylabel('Normalized cumulative distribution [NCD]')
plt.legend()
plt.title("Normalized cumulative distributions")
plt.show()

def determine_FWHM_axis(reference_axis, normalized_cumulative_distribution):
    # -1 sigma corresponds approximately to 0.15865 in the cumulative of a gaussian
    # +1 sigma corresponds to 0.84135
    NCD_index_left = np.argmin(np.abs(normalized_cumulative_distribution-0.15865))
    NCD_index_right = np.argmin(np.abs(normalized_cumulative_distribution-0.84135))

    # Fit small polynomials to get more precise estimates
    p_fitted_left = np.polynomial.Polynomial.fit(normalized_cumulative_distribution[NCD_index_left-1: NCD_index_left+2],
                                                 reference_axis[NCD_index_left-1: NCD_index_left+2],
                                                 deg=2)
    pixel_left = p_fitted_left(0.15865)

    p_fitted_right = np.polynomial.Polynomial.fit(normalized_cumulative_distribution[NCD_index_right-1: NCD_index_right+2],
                                                  reference_axis[NCD_index_right-1: NCD_index_right+2],
                                                  deg=2)
    pixel_right = p_fitted_right(0.84135)

    FWHM_factor = 2 * np.sqrt(2 * np.log(2)) # = 2.35482
    FWHM = (pixel_right - pixel_left)/2. * FWHM_factor

    return FWHM

FWHM_x = determine_FWHM_axis(X_axis, cumulative_sum_x)
FWHM_y = determine_FWHM_axis(Y_axis, cumulative_sum_y)

print('FWHM along the X axis: {0:.2f}'.format(FWHM_x))
print('FWHM along the Y axis: {0:.2f}'.format(FWHM_y))

# From the fits header of the first image (given):
# CCDSCALE= 0.25 arcsec/px unbinned
# BINX=4 and BINY=4 (4x4 binning)
seeing_x = FWHM_x * 4 * 0.25
seeing_y = FWHM_y * 4 * 0.25
print('Seeing along the X axis (after defocusing): {0:.2f} arcsec'.format(seeing_x))
print('Seeing along the Y axis (after defocusing): {0:.2f} arcsec'.format(seeing_y))

# End of the code
