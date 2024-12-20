"""
7 Differential Photometry

In this script, we perform differential photometry using aperture photometry results obtained
from a previously implemented AperturePhotometry class.



We also add comments to explain the changes and their purpose.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pickle
import matplotlib.colors as colors
from importlib import reload

from lecture06_aperture_photometry_with_classes import AperturePhotometry

taste_dir = 'TASTE_analysis/group05_QATAR-1_20230212'

# Load one science frame to visualize parameters
science_list = np.genfromtxt(f'{taste_dir}/science/science.list', dtype='str')
science_frame_name = f'{taste_dir}/correct/' + science_list[0][:-5] + '_corr.p'
science_corrected = pickle.load(open(science_frame_name, 'rb'))

def make_annulus_around_star(x_pos, y_pos, inner_radius, outer_radius, label='', color='y'):
    """
    Draws an annulus (a ring) around a star to visualize the sky background region.
    """
    from matplotlib.patches import Circle
    n, radii = 50, [inner_radius, outer_radius]
    theta = np.linspace(0, 2*np.pi, n, endpoint=True)
    xs = np.outer(radii, np.cos(theta))
    ys = np.outer(radii, np.sin(theta))
    xs[1,:] = xs[1,::-1]
    ys[1,:] = ys[1,::-1]
    ax.fill(np.ravel(xs)+x_pos, np.ravel(ys)+y_pos, edgecolor=None, facecolor=color, alpha=0.75, label=label)

def make_circle_around_star(x_pos, y_pos, radius, thickness=0.5, label='', color='w', alpha=1.):
    """
    Draws a circular aperture around a star.
    """
    from matplotlib.patches import Circle
    n, radii = 50, [radius, radius+thickness]
    theta = np.linspace(0, 2*np.pi, n, endpoint=True)
    xs = np.outer(radii, np.cos(theta))
    ys = np.outer(radii, np.sin(theta))
    xs[1,:] = xs[1,::-1]
    ys[1,:] = ys[1,::-1]
    ax.fill(np.ravel(xs)+x_pos, np.ravel(ys)+y_pos, edgecolor=None, facecolor=color, alpha=alpha, label=label)

# Coordinates and parameters from previous analysis
x_target = 416
y_target = 74

x_reference_01 = 298
y_reference_01 = 107

x_reference_02 = 117
y_reference_02 = 40

aperture = 8
inner_radius = 15
outer_radius = 23

vmin = np.amin(science_corrected[:,100:400])
vmax = np.amax(science_corrected[:,100:400])
print('vmin:  {0:.1f}    vmax: {1:.1f}'.format(vmin, vmax))
vmax = 5000

fig, ax = plt.subplots(1, figsize=(8,3))
im1 = plt.imshow(science_corrected, cmap=plt.colormaps['magma'],
                 norm=colors.LogNorm(vmin=vmin, vmax=vmax), origin='lower')
plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

make_circle_around_star(x_target, y_target, aperture, label='Aperture')
make_annulus_around_star(x_target, y_target, inner_radius, outer_radius, label='Target')
make_annulus_around_star(x_reference_01, y_reference_01, inner_radius, outer_radius, label='Reference #1', color='g')
make_annulus_around_star(x_reference_02, y_reference_02, inner_radius, outer_radius, label='Reference #2', color='r')

plt.xlabel(' X [pixels]')
plt.ylabel(' Y [pixels]')
ax.legend()
plt.show()

# Extracting aperture photometry for target and references
from time import time
t0 = time()

aperture = 8
inner_radius = 15
outer_radius = 23

target_ap08 = AperturePhotometry()
target_ap08.provide_aperture_parameters(inner_radius, outer_radius, aperture, x_target, y_target)
target_ap08.aperture_photometry()

reference01_ap08 = AperturePhotometry()
reference01_ap08.provide_aperture_parameters(inner_radius, outer_radius, aperture, x_reference_01, y_reference_01)
reference01_ap08.aperture_photometry()

reference02_ap08 = AperturePhotometry()
reference02_ap08.provide_aperture_parameters(inner_radius, outer_radius, aperture, x_reference_02, y_reference_02)
reference02_ap08.aperture_photometry()

t1 = time()
print('elapsed_time=', t1-t0)

time_offset = 2460024.0
normalization_index = 200

# IMPORTANT CORRECTION:
# Convert bjd_tdb (Time object) to numeric JD values before plotting
bjd_tdb_num = target_ap08.bjd_tdb.to_value('jd')
bjd_tdb_num_ref1 = reference01_ap08.bjd_tdb.to_value('jd')
bjd_tdb_num_ref2 = reference02_ap08.bjd_tdb.to_value('jd')

# Plot flux and other parameters
fig, axs = plt.subplots(5, 1, sharex=True, figsize=(8,10))
fig.subplots_adjust(hspace=0.05)

axs[0].scatter(bjd_tdb_num-time_offset,
               target_ap08.aperture/target_ap08.aperture[normalization_index],
               s=2, zorder=3, c='C0', label='Target')
axs[0].scatter(bjd_tdb_num-time_offset,
               reference01_ap08.aperture/reference01_ap08.aperture[normalization_index],
               s=2, zorder=2, c='C1', label='Ref #1')
axs[0].scatter(bjd_tdb_num-time_offset,
               reference02_ap08.aperture/reference02_ap08.aperture[normalization_index],
               s=2, zorder=1, c='C2', label='Ref #2')
axs[0].set_ylabel('Normalized flux')
axs[0].legend()

axs[1].scatter(bjd_tdb_num-time_offset, target_ap08.airmass, s=2, c='C0', label='Airmass')
axs[1].set_ylabel('Airmass')

axs[2].scatter(bjd_tdb_num-time_offset, target_ap08.sky_background, s=2, zorder=3, c='C0', label='Target')
axs[2].scatter(bjd_tdb_num-time_offset, reference01_ap08.sky_background, s=2, zorder=2, c='C1', label='Ref #1')
axs[2].scatter(bjd_tdb_num-time_offset, reference02_ap08.sky_background, s=2, zorder=1, c='C2', label='Ref #2')
axs[2].set_ylabel('Sky background [photons]')
axs[2].legend()

axs[3].scatter(bjd_tdb_num-time_offset, target_ap08.x_refined-target_ap08.x_refined[0], s=2, zorder=3, c='C0', label='X direction')
axs[3].scatter(bjd_tdb_num-time_offset, target_ap08.y_refined-target_ap08.y_refined[0], s=2, zorder=2, c='C1', label='Y direction')
axs[3].set_ylabel('Telescope drift [pixels]')
axs[3].legend()

axs[4].scatter(bjd_tdb_num-time_offset, target_ap08.x_fwhm, s=2, zorder=3, c='C0', label='X direction')
axs[4].scatter(bjd_tdb_num-time_offset, target_ap08.y_fwhm, s=2, zorder=2, c='C1', label='Y direction')
axs[4].set_ylabel('Target fWHM [pixels]')
axs[4].legend()

axs[4].set_xlabel('BJD-TDB - {0:.1f} [days]'.format(time_offset))
plt.show()

# Compute differential photometry
F_t = target_ap08.aperture
e_t = target_ap08.aperture_errors
F_r1 = reference01_ap08.aperture
e_r1 = reference01_ap08.aperture_errors
F_r2 = reference02_ap08.aperture
e_r2 = reference02_ap08.aperture_errors

differential_ap08_ref01 = F_t / F_r1
differential_ap08_ref02 = F_t / F_r2
F_rsum = F_r1 + F_r2
e_rsum = np.sqrt(e_r1**2 + e_r2**2)
differential_ap08_allref = F_t / F_rsum

# Error propagation for ratios:
# Se Q = A/B, allora sigma_Q = Q * sqrt((sigma_A/A)^2 + (sigma_B/B)^2)
differential_ap08_ref01_error = differential_ap08_ref01 * np.sqrt((e_t/F_t)**2 + (e_r1/F_r1)**2)
differential_ap08_ref02_error = differential_ap08_ref02 * np.sqrt((e_t/F_t)**2 + (e_r2/F_r2)**2)
differential_ap08_allref_error = differential_ap08_allref * np.sqrt((e_t/F_t)**2 + (e_rsum/F_rsum)**2)

# Plot differenziale con una singola stella di riferimento
plt.figure(figsize=(8,4))
plt.scatter(bjd_tdb_num - time_offset, differential_ap08_ref01, s=2, c='C0', label='Ref #1')
plt.xlabel('BJD-TDB - {0:.1f} [days]'.format(time_offset))
plt.ylabel('Differential photometry')
plt.ylim(2.200, 2.300)
plt.legend()
plt.show()

plt.figure(figsize=(8,4))
plt.scatter(bjd_tdb_num - time_offset, differential_ap08_ref02, s=2, c='C1', label='Ref #2')
plt.xlabel('BJD-TDB - {0:.1f} [days]'.format(time_offset))
plt.ylabel('Differential photometry')
plt.ylim(0.268, 0.278)
plt.legend()
plt.show()

plt.figure(figsize=(8,4))
plt.scatter(bjd_tdb_num - time_offset, differential_ap08_allref, s=2, c='C3', label='Sum of refs')
plt.xlabel('BJD-TDB - {0:.1f} [days]'.format(time_offset))
plt.ylabel('Differential photometry')
plt.ylim(0.240, 0.250)
plt.legend()
plt.show()

# Selezione della regione out-of-transit:
out_transit_selection = ((bjd_tdb_num < 2460024.3450) | (bjd_tdb_num > 2460024.4350)) & (bjd_tdb_num < 2460024.4475)

# Fit polinomiale per normalizzare la curva
# Convertiamo i tempi in bjd_tdb_num se non l'abbiamo già fatto sopra
bjd_median = np.median(bjd_tdb_num)

from numpy.polynomial import Polynomial

poly_ap08_ref01_deg01_pfit = Polynomial.fit(bjd_tdb_num[out_transit_selection]-bjd_median, differential_ap08_ref01[out_transit_selection], deg=1)
poly_ap08_ref02_deg01_pfit = Polynomial.fit(bjd_tdb_num[out_transit_selection]-bjd_median, differential_ap08_ref02[out_transit_selection], deg=1)
poly_ap08_allref_deg01_pfit = Polynomial.fit(bjd_tdb_num[out_transit_selection]-bjd_median, differential_ap08_allref[out_transit_selection], deg=1)

differential_ap08_ref01_normalized = differential_ap08_ref01 / poly_ap08_ref01_deg01_pfit(bjd_tdb_num - bjd_median)
differential_ap08_ref02_normalized = differential_ap08_ref02 / poly_ap08_ref02_deg01_pfit(bjd_tdb_num - bjd_median)
differential_ap08_allref_normalized = differential_ap08_allref / poly_ap08_allref_deg01_pfit(bjd_tdb_num - bjd_median)

# Propagazione errore anche per la versione normalizzata:
differential_ap08_ref01_normalized_error = differential_ap08_ref01_error / poly_ap08_ref01_deg01_pfit(bjd_tdb_num - bjd_median)
differential_ap08_ref02_normalized_error = differential_ap08_ref02_error / poly_ap08_ref02_deg01_pfit(bjd_tdb_num - bjd_median)
differential_ap08_allref_normalized_error = differential_ap08_allref_error / poly_ap08_allref_deg01_pfit(bjd_tdb_num - bjd_median)

plt.figure()
plt.scatter(bjd_tdb_num-time_offset, differential_ap08_ref01_normalized, s=2)
plt.scatter(bjd_tdb_num-time_offset, differential_ap08_ref02_normalized, s=2)
plt.scatter(bjd_tdb_num-time_offset, differential_ap08_allref_normalized, s=2)

plt.axvline(2460024.3450-time_offset, c='C3')
plt.axvline(2460024.4350-time_offset, c='C3')
plt.xlim(0.3, 0.5)
plt.ylim(0.975, 1.025)

plt.xlabel('BJD-TDB - {0:.1f} [days]'.format(time_offset))
plt.ylabel('Normalized differential photometry')
plt.show()

print('Standard deviation aperture 08 reference #1:    {0:.7f}'.format(np.std(differential_ap08_ref01_normalized[out_transit_selection])))
print('Standard deviation aperture 08 reference #2:    {0:.7f}'.format(np.std(differential_ap08_ref02_normalized[out_transit_selection])))
print('Standard deviation aperture 08 all references : {0:.7f}'.format(np.std(differential_ap08_allref_normalized[out_transit_selection])))

# Salvataggio dei risultati
pickle.dump(bjd_tdb_num, open('taste_bjdtdb.p','wb'))
pickle.dump(differential_ap08_ref01_normalized, open('differential_ap08_ref01_normalized.p','wb'))
pickle.dump(differential_ap08_ref01_normalized_error, open('differential_ap08_ref01_normalized_error.p','wb'))
pickle.dump(differential_ap08_allref_normalized, open('differential_ap08_allref_normalized.p','wb'))
pickle.dump(differential_ap08_allref_normalized_error, open('differential_ap08_allref_normalized_error.p','wb'))

pickle.dump(differential_ap08_ref01, open('differential_ap08_ref01.p','wb'))
pickle.dump(differential_ap08_ref01_error, open('differential_ap08_ref01_error.p','wb'))
pickle.dump(differential_ap08_allref, open('differential_ap08_allref.p','wb'))
pickle.dump(differential_ap08_allref_error, open('differential_ap08_allref_error.p','wb'))

