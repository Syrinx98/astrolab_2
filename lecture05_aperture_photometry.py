"""
4 Sky background subtraction and aperture photometry
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle
from astropy.io import fits
import time
from matplotlib import colors
from scipy.stats import multivariate_normal

from lecture01_bias_analysis import taste_dir

# =============================================================================
print("\n\n 2.1. Read a science frame and its associated errors")
print("=============================================================================\n")

"""
Sky background subtraction and aperture photometry

We load one scientific frame and the corresponding error estimates.
"""
taste_dir = 'TASTE_analysis/group05_QATAR-1_20230212'

science_list = np.genfromtxt(f'{taste_dir}/science/science.list', dtype='str')
science_temp_list = science_list[:10]

science_corrected = pickle.load(open(f'{taste_dir}/correct/' + science_temp_list[5][:-5] + '_corr.p', 'rb'))
science_corrected_err = pickle.load(open(f'{taste_dir}/correct/' + science_temp_list[5][:-5] + '_corr_errors.p', 'rb'))

# =============================================================================
print("\n\n 2.2. Compute distance arrays and identify target coordinates")
print("=============================================================================\n")

"""
We have the refined coordinates of the target from the previous analysis step.
We also build meshgrid arrays and compute distances from the target.
"""

x_target_refined = 415.851
y_target_refined = 73.959

ylen, xlen = np.shape(science_corrected)
X_axis = np.arange(0, xlen, 1)
Y_axis = np.arange(0, ylen, 1)
X, Y = np.meshgrid(X_axis, Y_axis)

target_distance = np.sqrt((X - x_target_refined)**2 + (Y - y_target_refined)**2)

# =============================================================================
print("\n\n 2.3. Function to draw circles on the image")
print("=============================================================================\n")

"""
Define a function to plot circles.
"""

def make_circle_around_star(x_pos, y_pos, radius, thickness=0.5, label='', color='w', alpha=1.):
    from matplotlib.patches import Circle
    n, radii = 50, [radius, radius+thickness]
    theta = np.linspace(0, 2*np.pi, n, endpoint=True)
    xs = np.outer(radii, np.cos(theta))
    ys = np.outer(radii, np.sin(theta))
    xs[1,:] = xs[1,::-1]
    ys[1,:] = ys[1,::-1]
    ax.fill(np.ravel(xs)+x_pos, np.ravel(ys)+y_pos, edgecolor=None, facecolor=color, alpha=alpha, label=label)


# =============================================================================
print("\n\n 2.4. Selecting the annulus for sky background measurement")
print("=============================================================================\n")

"""
We define inner and outer radii for the annulus around the star to measure the sky background.
Plot the region around the star with a suitable vmax to highlight the background.
"""

inner_radius = 15
outer_radius = 22

vmin = np.amin(science_corrected[:,100:400])
vmax = np.amax(science_corrected[:,100:400])
vmax = 2*vmin
print('vmin:  {0:.1f}    vmax: {1:.1f}'.format(vmin, vmax))

fig, ax = plt.subplots(1, figsize=(5,5))
im1 = plt.imshow(science_corrected, cmap=plt.colormaps['magma'],
                 norm=colors.LogNorm(vmin=vmin, vmax=vmax), origin='lower')
plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
plt.xlim(x_target_refined - outer_radius*1.2, x_target_refined + outer_radius*1.2)
plt.ylim(y_target_refined - outer_radius*1.2, y_target_refined + outer_radius*1.2)

make_circle_around_star(x_target_refined, y_target_refined, inner_radius, label='inner radius')
make_circle_around_star(x_target_refined, y_target_refined, outer_radius, color='y', label='outer radius')

plt.xlabel(' X [pixels]')
plt.ylabel(' Y [pixels]')
plt.legend(loc='upper left')
plt.title("Annulus selection for sky background")
plt.show()


# =============================================================================
print("\n\n 2.5. Compute the sky background and median")
print("=============================================================================\n")

"""
Select pixels between inner_radius and outer_radius to compute sky flux average/median.
We ensure that enough pixels are included in the annulus.
"""

annulus_selection = (target_distance > inner_radius) & (target_distance <= outer_radius)

sky_flux_average = np.sum(science_corrected[annulus_selection]) / np.sum(annulus_selection)
sky_flux_median = np.median(science_corrected[annulus_selection])

print('Number of pixels included in the annulus: {0:7.0f}'.format(np.sum(annulus_selection)))
print('Average Sky flux: {0:7.1f} photons/pixel'.format(sky_flux_average))
print('Median Sky flux:  {0:7.1f} photons/pixel'.format(sky_flux_median))

# =============================================================================
print("\n\n HOMEWORK: Compute the associated error to the sky background")
print("=============================================================================\n")

"""
HOMEWORK:
1) Compute the associated error to the sky background, remembering that each pixel has an associated error.
2) Check if median and average are consistent within errors.
3) Decide if using the median or average makes a difference.
"""

# Example (not a final solution, just a placeholder comment):
# sky_flux_error = np.sqrt( np.sum(science_corrected_err[annulus_selection]**2) ) / np.sum(annulus_selection)
# print('Sky flux error (example): {0:7.1f} photons/pixel'.format(sky_flux_error))


# =============================================================================
print("\n\n 2.6. Remove the sky background from the measured flux of the star")
print("=============================================================================\n")

"""
We can now subtract the sky background from the entire frame.
HOMEWORK:
Compute the error associated to the flux on each pixel after subtracting the sky level.
"""

science_sky_corrected = science_corrected - sky_flux_average
# science_sky_corrected_error = np.sqrt(science_corrected_err**2 + sky_flux_error^2) # example if sky error known


# =============================================================================
print("\n\n 2.7. Aperture photometry")
print("=============================================================================\n")

"""
We measure the flux at different apertures. We define an aperture radius and sum up the flux inside it.
We compare the flux at various radii to decide which aperture to use.
"""

inner_selection = (target_distance < inner_radius)
total_flux = np.sum(science_sky_corrected[inner_selection])

radius_array = np.arange(0, inner_radius + 1., 0.5)
flux_vs_radius = np.zeros_like(radius_array)

for ii, aperture_radius in enumerate(radius_array):
    aperture_selection = (target_distance < aperture_radius)
    flux_vs_radius[ii] = np.sum(science_sky_corrected[aperture_selection])/total_flux

plt.figure()
plt.scatter(radius_array, flux_vs_radius, c='C1')
plt.axhline(0.80)
plt.axhline(0.85)
plt.axhline(0.90)
plt.axhline(0.95)
plt.axhline(0.99)

plt.xlabel('Aperture [pixels]')
plt.ylabel('Fractional flux within the aperture')
plt.title("Fraction of flux captured as a function of aperture radius")
plt.show()

"""
HOMEWORK:
1) Identify at least two apertures for photometric analysis.
2) Compute the error associated with each flux measurement.
Remember: here you are summing fluxes, not taking an average. The error will propagate accordingly.
"""

# =============================================================================
print("\n\n 2.8. Comparing the star's profile with a multivariate normal distribution")
print("=============================================================================\n")

"""
To get a qualitative idea of how close or far the star PSF is from a 2D Gaussian,
we compare the measured fractional flux vs radius with that of simulated multivariate normal distributions.
"""

xy_range = np.arange(-inner_radius*1.2, inner_radius*1.2, 0.1)
X_gauss, Y_gauss = np.meshgrid(xy_range, xy_range)
pos = np.dstack((X_gauss, Y_gauss))
gauss_distance = np.sqrt(X_gauss**2 + Y_gauss**2)

plot_range = np.arange(0, inner_radius, 0.1)

mv_normal_cov05 = multivariate_normal(mean=[0.,0.], cov=5., allow_singular=False)
mv_normal_cov05_pdf = mv_normal_cov05.pdf(pos)
plot_cov05_flux = np.zeros_like(plot_range)

mv_normal_cov10 = multivariate_normal(mean=[0.,0.], cov=10., allow_singular=False)
mv_normal_cov10_pdf = mv_normal_cov10.pdf(pos)
plot_cov10_flux = np.zeros_like(plot_range)

mv_normal_cov20 = multivariate_normal(mean=[0.,0.], cov=20., allow_singular=False)
mv_normal_cov20_pdf = mv_normal_cov20.pdf(pos)
plot_cov20_flux = np.zeros_like(plot_range)

for ii, aperture_radius in enumerate(plot_range):
    pdf_selection = (gauss_distance < aperture_radius)
    plot_cov05_flux[ii] = np.sum(mv_normal_cov05_pdf[pdf_selection])
    plot_cov10_flux[ii] = np.sum(mv_normal_cov10_pdf[pdf_selection])
    plot_cov20_flux[ii] = np.sum(mv_normal_cov20_pdf[pdf_selection])

plt.figure()
plt.scatter(radius_array, flux_vs_radius, c='C1', label='Measurements')
plt.plot(plot_range, plot_cov05_flux/plot_cov05_flux[-1], c='C2', label='Covariance=5.')
plt.plot(plot_range, plot_cov10_flux/plot_cov10_flux[-1], c='C3', label='Covariance=10.')
plt.plot(plot_range, plot_cov20_flux/plot_cov20_flux[-1], c='C4', label='Covariance=20.')

plt.xlabel('Aperture [pixels]')
plt.ylabel('Fractional flux within the aperture')
plt.legend()
plt.title("Comparison with multivariate normal distributions")
plt.show()
