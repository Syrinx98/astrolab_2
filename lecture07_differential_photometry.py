#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle
from astropy.io import fits
from astropy import coordinates as coord, units as u
from astropy.time import Time

from lecture06_aperture_photometry_with_classes import AperturePhotometry


##############################################################################
# 2) Optional helper functions to plot apertures/annuli on a single frame
##############################################################################

def make_annulus_around_star(ax, x_pos, y_pos, inner_radius, outer_radius, label='', color='y'):
    """
    Draw an annulus on the existing axis `ax`.
    """
    from matplotlib.patches import Circle
    import matplotlib.path as mpath
    import matplotlib.patches as mpatches

    n = 50
    theta = np.linspace(0, 2*np.pi, n, endpoint=True)
    # two circles: inner, outer
    r_in, r_out = inner_radius, outer_radius

    x_inner = x_pos + r_in * np.cos(theta)
    y_inner = y_pos + r_in * np.sin(theta)
    x_outer = x_pos + r_out * np.cos(theta)
    y_outer = y_pos + r_out * np.sin(theta)

    # Reverse the inner circle so that it goes in the opposite direction
    x_inner = x_inner[::-1]
    y_inner = y_inner[::-1]

    coords = np.vstack((np.append(x_outer, x_inner), np.append(y_outer, y_inner))).T
    path = mpath.Path(coords)
    patch = mpatches.PathPatch(path, facecolor=color, alpha=0.4, label=label)
    ax.add_patch(patch)

def make_circle_around_star(ax, x_pos, y_pos, radius, thickness=0.5, label='', color='w', alpha=1.):
    """
    Draw a circular region on the existing axis `ax`.
    """
    from matplotlib.patches import Circle
    import matplotlib.path as mpath
    import matplotlib.patches as mpatches

    n = 50
    theta = np.linspace(0, 2*np.pi, n, endpoint=True)
    r_in, r_out = radius, radius + thickness

    x_inner = x_pos + r_in * np.cos(theta)
    y_inner = y_pos + r_in * np.sin(theta)
    x_outer = x_pos + r_out * np.cos(theta)
    y_outer = y_pos + r_out * np.sin(theta)

    # Reverse the inner circle so that it goes in the opposite direction
    x_inner = x_inner[::-1]
    y_inner = y_inner[::-1]

    coords = np.vstack((np.append(x_outer, x_inner), np.append(y_outer, y_inner))).T
    path = mpath.Path(coords)
    patch = mpatches.PathPatch(path, facecolor=color, alpha=alpha, label=label)
    ax.add_patch(patch)

##############################################################################
# 3) Main code demonstrating usage
##############################################################################

def main():
    taste_dir = 'TASTE_analysis/group05_QATAR-1_20230212'
    ##########################################################################
    # A) Show an example corrected science frame with the aperture/annulus
    ##########################################################################
    # For demonstration, let's load the first corrected frame
    # (You need to have your own pickled corrected science frames
    #  or adapt to read the raw FITS + calibrations. This snippet
    #  loads from a file called  <some_frame>_corr.p  as in your text.)

    # For the example, let's just load from the 'correct' folder
    science_list = np.genfromtxt(f'{taste_dir}/science/science.list', dtype=str)
    first_frame_name = f'{taste_dir}/correct/' + science_list[0][:-5] + '_corr.p'
    science_corrected = pickle.load(open(first_frame_name, 'rb'))  # 2D array (floats)

    # Coordinates and photometric parameters
    x_target = 416
    y_target = 74

    x_reference_01 = 298
    y_reference_01 = 107

    x_reference_02 = 117
    y_reference_02 = 40

    aperture = 8
    inner_radius = 10
    outer_radius = 18

    # Show the image and the star apertures
    fig, ax = plt.subplots(1, figsize=(8, 4), dpi=300)
    vmin = np.amin(science_corrected[:, 100:400])
    vmax = np.amax(science_corrected[:, 100:400])
    vmax = 5000  # example to saturate the scale a bit

    im1 = ax.imshow(science_corrected, cmap='magma',
                    norm=colors.LogNorm(vmin=vmin, vmax=vmax), origin='lower')
    plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

    make_circle_around_star(ax, x_target, y_target, aperture, label='Target Aperture')
    make_annulus_around_star(ax, x_target, y_target, inner_radius, outer_radius, label='Target Annulus', color='y')
    make_annulus_around_star(ax, x_reference_01, y_reference_01, inner_radius, outer_radius,
                             label='Reference #1 Annulus', color='g')
    make_annulus_around_star(ax, x_reference_02, y_reference_02, inner_radius, outer_radius,
                             label='Reference #2 Annulus', color='r')

    ax.set_xlabel('X [pixels]')
    ax.set_ylabel('Y [pixels]')
    ax.legend()
    plt.title("Example science frame with aperture & annulus")
    plt.show()

    ##########################################################################
    # B) Perform AperturePhotometry for the target and references
    ##########################################################################
    from time import time
    t0 = time()

    # Build the objects
    target = AperturePhotometry()
    target.provide_aperture_parameters(inner_radius, outer_radius, aperture, x_target, y_target)
    target.aperture_photometry()

    reference01 = AperturePhotometry()
    reference01.provide_aperture_parameters(inner_radius, outer_radius, aperture, x_reference_01, y_reference_01)
    reference01.aperture_photometry()

    reference02 = AperturePhotometry()
    reference02.provide_aperture_parameters(inner_radius, outer_radius, aperture, x_reference_02, y_reference_02)
    reference02.aperture_photometry()

    t1 = time()
    print('Aperture photometry completed in {0:.2f} seconds.'.format(t1-t0))

    # For convenience, define a short variable for time
    bjd_tdb = target.bjd_tdb
    time_offset = 2460024.0  # example offset to have smaller numbers on the x axis
    normalization_index = 29  # an index for normalizing flux

    ##########################################################################
    # C) Plot raw flux, airmass, sky background, drift, FWHM
    ##########################################################################
    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(8,10))
    fig.subplots_adjust(hspace=0.05)

    # 1) Flux
    axs[0].scatter(bjd_tdb.value - time_offset,
                   target.aperture/target.aperture[normalization_index],
                   s=3, label='Target')
    axs[0].scatter(bjd_tdb.value - time_offset,
                   reference01.aperture/reference01.aperture[normalization_index],
                   s=3, label='Ref #1')
    axs[0].scatter(bjd_tdb.value - time_offset,
                   reference02.aperture/reference02.aperture[normalization_index],
                   s=3, label='Ref #2')
    axs[0].legend()
    axs[0].set_ylabel('Normalized flux')

    # 2) Airmass
    axs[1].scatter(bjd_tdb.value - time_offset, target.airmass, s=3, c='C0', label='Airmass')
    axs[1].set_ylabel('Airmass')

    # 3) Sky background
    axs[2].scatter(bjd_tdb.value - time_offset, target.sky_background, s=3, label='Target')
    axs[2].scatter(bjd_tdb.value - time_offset, reference01.sky_background, s=3, label='Ref #1')
    axs[2].scatter(bjd_tdb.value - time_offset, reference02.sky_background, s=3, label='Ref #2')
    axs[2].set_ylabel('Sky background [photons]')
    axs[2].legend()

    # 4) Telescope drift
    axs[3].scatter(bjd_tdb.value - time_offset, target.x_refined - target.x_refined[0],
                   s=3, label='X drift')
    axs[3].scatter(bjd_tdb.value - time_offset, target.y_refined - target.y_refined[0],
                   s=3, label='Y drift')
    axs[3].legend()
    axs[3].set_ylabel('Drift [pix]')

    # 5) FWHM
    axs[4].scatter(bjd_tdb.value - time_offset, target.x_fwhm, s=3, label='X FWHM')
    axs[4].scatter(bjd_tdb.value - time_offset, target.y_fwhm, s=3, label='Y FWHM')
    axs[4].legend()
    axs[4].set_ylabel('FWHM [pix]')

    axs[-1].set_xlabel(f'BJD_TDB - {time_offset:.1f} [days]')
    plt.suptitle("Raw flux and diagnostic plots (Aperture=8 pix)")
    plt.show()

    ##########################################################################
    # D) Compute differential photometry + Propagation of errors
    ##########################################################################
    #  Single references:
    diff_ref01 = target.aperture / reference01.aperture
    diff_ref02 = target.aperture / reference02.aperture

    #  Errors in ratio = ratio * sqrt( (err_target/target)^2 + (err_ref/ref)^2 )
    diff_ref01_err = diff_ref01 * np.sqrt(
        (target.aperture_errors/target.aperture)**2 +
        (reference01.aperture_errors/reference01.aperture)**2
    )
    diff_ref02_err = diff_ref02 * np.sqrt(
        (target.aperture_errors/target.aperture)**2 +
        (reference02.aperture_errors/reference02.aperture)**2
    )

    #  Sum of references:
    sum_refs = reference01.aperture + reference02.aperture
    sum_refs_err = np.sqrt(reference01.aperture_errors**2 + reference02.aperture_errors**2)
    diff_allref = target.aperture / sum_refs
    diff_allref_err = diff_allref * np.sqrt(
        (target.aperture_errors/target.aperture)**2 +
        (sum_refs_err/sum_refs)**2
    )

    # Quick plot
    plt.figure(figsize=(8,4))
    plt.scatter(bjd_tdb.value - time_offset, diff_ref01, s=2, label='Ref #1')
    plt.scatter(bjd_tdb.value - time_offset, diff_ref02, s=2, label='Ref #2')
    plt.scatter(bjd_tdb.value - time_offset, diff_allref, s=2, label='All Refs')
    plt.xlabel(f'BJD_TDB - {time_offset:.1f} [days]')
    plt.ylabel('Differential photometry (raw ratio)')
    plt.legend()
    plt.show()

    ##########################################################################
    # E) Fit a polynomial trend and normalize
    ##########################################################################
    # out-of-transit region: for example, exclude from 2460024.3450 -> 2460024.4350
    out_transit_sel = ((bjd_tdb.value < 2460024.3450) | (bjd_tdb.value > 2460024.4350))

    from numpy.polynomial import Polynomial
    bjd_median = np.median(bjd_tdb.value)

    # Fit linear polynomials
    pfit_ref01 = Polynomial.fit(bjd_tdb.value[out_transit_sel] - bjd_median,
                                diff_ref01[out_transit_sel], deg=1)
    pfit_ref02 = Polynomial.fit(bjd_tdb.value[out_transit_sel] - bjd_median,
                                diff_ref02[out_transit_sel], deg=1)
    pfit_all   = Polynomial.fit(bjd_tdb.value[out_transit_sel] - bjd_median,
                                diff_allref[out_transit_sel], deg=1)

    # Evaluate the polynomials -> normalization factor
    norm_ref01 = pfit_ref01(bjd_tdb.value - bjd_median)
    norm_ref02 = pfit_ref02(bjd_tdb.value - bjd_median)
    norm_all   = pfit_all(bjd_tdb.value - bjd_median)

    diff_ref01_norm = diff_ref01 / norm_ref01
    diff_ref02_norm = diff_ref02 / norm_ref02
    diff_allref_norm = diff_allref / norm_all

    # Propagate errors for normalized curve:
    #   (target/reference)/fitted_trend -> total factor is 1 / fitted_trend
    #   so error ~ sqrt( (diff_err/fitted_trend)^2 + (diff * trend_err / trend^2 )^2 ) ...
    #   but we have not derived a formal error on the polynomial.
    # Here we just do the simplest approach ignoring polynomial fit error:
    diff_ref01_norm_err = diff_ref01_err / norm_ref01
    diff_ref02_norm_err = diff_ref02_err / norm_ref02
    diff_allref_norm_err = diff_allref_err / norm_all

    # Plot the normalized differential photometry
    plt.figure()
    plt.scatter(bjd_tdb.value, diff_ref01_norm, s=2, label='Ref #1 norm')
    plt.scatter(bjd_tdb.value, diff_ref02_norm, s=2, label='Ref #2 norm')
    plt.scatter(bjd_tdb.value, diff_allref_norm, s=2, label='All refs norm')

    # 2 red lines wrt to eh BJD median
    plt.axvline(x=bjd_median - 0.5*(np.max(bjd_tdb.value) - bjd_median), c='C3')
    plt.axvline(x=bjd_median + 0.5*(np.max(bjd_tdb.value) - bjd_median), c='C3')
    plt.autoscale()

    plt.ylim(0.95, 1.05)
    plt.xlabel(f'BJD_TDB - {time_offset:.1f} [days]')
    plt.ylabel('Normalized differential photometry')
    plt.legend()
    plt.show()

    # Print standard deviation outside transit
    std_ref01 = np.std(diff_ref01_norm[out_transit_sel])
    std_ref02 = np.std(diff_ref02_norm[out_transit_sel])
    std_all   = np.std(diff_allref_norm[out_transit_sel])
    print(f'Standard deviation (out of transit) Ref01: {std_ref01:.6f}')
    print(f'Standard deviation (out of transit) Ref02: {std_ref02:.6f}')
    print(f'Standard deviation (out of transit) AllRefs: {std_all:.6f}')

    ##########################################################################
    # F) Save some results via pickle
    ##########################################################################
    # Example: saving the BJD, the ratio with errors, etc.
    pickle.dump(bjd_tdb.value, open(f'{taste_dir}/taste_bjdtdb.p', 'wb'))

    # Save the raw ratio
    pickle.dump(diff_allref,         open(f'{taste_dir}/differential_allref.p','wb'))
    pickle.dump(diff_allref_err,     open(f'{taste_dir}/differential_allref_error.p','wb'))

    # Save the normalized ratio
    pickle.dump(diff_allref_norm,    open(f'{taste_dir}/differential_allref_normalized.p','wb'))
    pickle.dump(diff_allref_norm_err,open(f'{taste_dir}/differential_allref_normalized_error.p','wb'))

    print("Done. Data saved to pickle files.")

##############################################################################
# 4) Run everything
##############################################################################
if __name__ == '__main__':
    main()
