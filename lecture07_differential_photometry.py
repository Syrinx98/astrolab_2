#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved script for Lecture 07: Differential Photometry
 - Reads corrected science frames
 - Displays apertures/annuli on an example frame
 - Performs aperture photometry on target and two reference stars
 - Plots diagnostics: raw flux, airmass, sky background, drift, FWHM
 - Computes differential photometry with error propagation
 - Fits and removes polynomial trends; normalizes light curves
 - Computes and prints out-of-transit scatter
 - Saves results to pickle files
 - Explores performance for different aperture radii

KEEP ALL FILE PATHS EXACT AS ORIGINAL (taste_dir, etc.)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle
from astropy.io import fits
from astropy import coordinates as coord, units as u
from astropy.time import Time
from lecture06_aperture_photometry_with_classes import AperturePhotometry
from numpy.polynomial import Polynomial


def make_annulus_around_star(ax, x_pos, y_pos, inner_radius, outer_radius,
                             label='', color='y', alpha=0.4):
    """
    Draw a transparent annulus (sky region) on axis `ax` around (x_pos, y_pos).
    """
    from matplotlib import patches, path
    theta = np.linspace(0, 2*np.pi, 100)
    x_outer = x_pos + outer_radius * np.cos(theta)
    y_outer = y_pos + outer_radius * np.sin(theta)
    x_inner = x_pos + inner_radius * np.cos(theta)[::-1]
    y_inner = y_pos + inner_radius * np.sin(theta)[::-1]
    coords = np.vstack((np.column_stack((x_outer, y_outer)),
                        np.column_stack((x_inner, y_inner))))
    ann = patches.PathPatch(path.Path(coords), facecolor=color,
                             edgecolor=None, alpha=alpha, label=label)
    ax.add_patch(ann)


def make_circle_around_star(ax, x_pos, y_pos, radius, thickness=0.5,
                            label='', color='w', alpha=1.0):
    """
    Draw a filled circular aperture on axis `ax` around (x_pos, y_pos).
    """
    from matplotlib import patches, path
    theta = np.linspace(0, 2*np.pi, 100)
    x_outer = x_pos + (radius + thickness) * np.cos(theta)
    y_outer = y_pos + (radius + thickness) * np.sin(theta)
    x_inner = x_pos + radius * np.cos(theta)[::-1]
    y_inner = y_pos + radius * np.sin(theta)[::-1]
    coords = np.vstack((np.column_stack((x_outer, y_outer)),
                        np.column_stack((x_inner, y_inner))))
    circ = patches.PathPatch(path.Path(coords), facecolor=color,
                             edgecolor=None, alpha=alpha, label=label)
    ax.add_patch(circ)


def main():
    # Base directory for this group's TASTE analysis
    taste_dir = 'TASTE_analysis/group05_QATAR-1_20230212'

    # --------------------------------------------------------------
    # A) Load and display the first corrected science frame
    # --------------------------------------------------------------
    print("Loading science list from:", f"{taste_dir}/science/science.list")
    science_list = np.genfromtxt(f'{taste_dir}/science/science.list', dtype=str)
    first_basename = science_list[0][:-5]
    first_frame_file = f'{taste_dir}/correct/{first_basename}_corr.p'
    print(f"Loading first corrected frame: {first_frame_file}")
    science_corrected = pickle.load(open(first_frame_file, 'rb'))

    # Photometry settings and star positions
    x_target, y_target = 416, 74
    x_ref1, y_ref1 = 298, 107
    x_ref2, y_ref2 = 117, 40
    aperture = 8
    inner_radius = 10
    outer_radius = 18

    print("Displaying example frame with apertures and annuli...")
    fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
    pos = science_corrected[np.isfinite(science_corrected) & (science_corrected > 0)]
    vmin = np.percentile(pos, 5) if pos.size else np.nanmin(science_corrected)
    vmax = np.percentile(pos, 99) if pos.size else np.nanmax(science_corrected)
    if vmin <= 0 or vmin >= vmax:
        vmin, vmax = pos.min(), pos.max()
    try:
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    except ValueError:
        print("Warning: Invalid LogNorm; using linear scaling")
        norm = None
    im = ax.imshow(science_corrected, cmap='magma', norm=norm, origin='lower')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pixel value' + (' (log scale)' if norm else ''))

    make_circle_around_star(ax, x_target, y_target, aperture,
                            label='Target Aperture', color='w', alpha=0.7)
    make_annulus_around_star(ax, x_target, y_target,
                             inner_radius, outer_radius,
                             label='Target Annulus', color='y')
    make_annulus_around_star(ax, x_ref1, y_ref1,
                             inner_radius, outer_radius,
                             label='Ref #1 Annulus', color='g')
    make_annulus_around_star(ax, x_ref2, y_ref2,
                             inner_radius, outer_radius,
                             label='Ref #2 Annulus', color='r')

    ax.set_xlabel('X [pixels]')
    ax.set_ylabel('Y [pixels]')
    ax.legend(loc='upper right', fontsize='small')
    ax.set_title('Example science frame with aperture & annuli')
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------------
    # B) Perform aperture photometry
    # --------------------------------------------------------------
    print("Running aperture photometry on target and references...")
    from time import time
    t0 = time()
    target = AperturePhotometry()
    target.provide_aperture_parameters(inner_radius, outer_radius,
                                       aperture, x_target, y_target)
    target.aperture_photometry()

    ref1 = AperturePhotometry()
    ref1.provide_aperture_parameters(inner_radius, outer_radius,
                                     aperture, x_ref1, y_ref1)
    ref1.aperture_photometry()

    ref2 = AperturePhotometry()
    ref2.provide_aperture_parameters(inner_radius, outer_radius,
                                     aperture, x_ref2, y_ref2)
    ref2.aperture_photometry()

    print(f"Aperture photometry done in {time()-t0:.2f} s")

    bjd = target.bjd_tdb.value
    offset = 2460024.0
    norm_idx = 29

    # --------------------------------------------------------------
    # C) Diagnostic plots
    # --------------------------------------------------------------
    print("Generating diagnostics...")
    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(8, 10), dpi=150)
    fig.subplots_adjust(hspace=0.05)

    # Flux
    axs[0].scatter(bjd-offset, target.aperture/target.aperture[norm_idx], s=5, label='Target')
    axs[0].scatter(bjd-offset, ref1.aperture/ref1.aperture[norm_idx], s=5, label='Ref #1')
    axs[0].scatter(bjd-offset, ref2.aperture/ref2.aperture[norm_idx], s=5, label='Ref #2')
    axs[0].set_ylabel('Normalized Flux'); axs[0].legend(fontsize='small')

    # Airmass
    axs[1].scatter(bjd-offset, target.airmass, s=5); axs[1].set_ylabel('Airmass')

    # Sky
    axs[2].scatter(bjd-offset, target.sky_background, s=5, label='Target')
    axs[2].scatter(bjd-offset, ref1.sky_background, s=5, label='Ref #1')
    axs[2].scatter(bjd-offset, ref2.sky_background, s=5, label='Ref #2')
    axs[2].set_ylabel('Sky [photons]'); axs[2].legend(fontsize='small')

    # Drift
    axs[3].scatter(bjd-offset, target.x_position-target.x_position[0], s=5, label='X drift')
    axs[3].scatter(bjd-offset, target.y_position-target.y_position[0], s=5, label='Y drift')
    axs[3].set_ylabel('Drift [pix]'); axs[3].legend(fontsize='small')

    # FWHM
    axs[4].scatter(bjd-offset, target.x_fwhm, s=5, label='X FWHM')
    axs[4].scatter(bjd-offset, target.y_fwhm, s=5, label='Y FWHM')
    axs[4].set_ylabel('FWHM [pix]'); axs[4].legend(fontsize='small')
    axs[4].set_xlabel(f'BJD_TDB - {offset:.1f} [days]')
    plt.suptitle('Diagnostics: Flux, Airmass, Sky, Drift, FWHM', y=0.93)
    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.show()

    # --------------------------------------------------------------
    # D) Differential photometry + errors
    # --------------------------------------------------------------
    print("Computing differential photometry and errors…")
    diff1 = target.aperture / ref1.aperture
    diff2 = target.aperture / ref2.aperture
    sum_ref = ref1.aperture + ref2.aperture
    diff_all = target.aperture / sum_ref
    err1 = diff1 * np.sqrt((target.aperture_errors/target.aperture)**2 +
                           (ref1.aperture_errors/ref1.aperture)**2)
    err2 = diff2 * np.sqrt((target.aperture_errors/target.aperture)**2 +
                           (ref2.aperture_errors/ref2.aperture)**2)
    sum_err = np.sqrt(ref1.aperture_errors**2 + ref2.aperture_errors**2)
    err_all = diff_all * np.sqrt((target.aperture_errors/target.aperture)**2 +
                                  (sum_err/sum_ref)**2)

    # D′.1: Ref #1
    plt.figure(figsize=(8,4), dpi=150)
    plt.scatter(bjd-offset, diff1, s=4, label='Ref #1')
    plt.xlabel(f'BJD_TDB - {offset:.1f} [d]')
    plt.ylabel('Differential flux (target/ref1)')
    plt.legend(fontsize='small')
    plt.tight_layout()
    plt.show()

    # D′.2: Ref #2
    plt.figure(figsize=(8,4), dpi=150)
    plt.scatter(bjd-offset, diff2, s=4, label='Ref #2', color='C1')
    plt.xlabel(f'BJD_TDB - {offset:.1f} [d]')
    plt.ylabel('Differential flux (target/ref2)')
    plt.legend(fontsize='small')
    plt.tight_layout()
    plt.show()

    # D′.3: Sum of refs
    plt.figure(figsize=(8,4), dpi=150)
    plt.scatter(bjd-offset, diff_all, s=4, label='Sum of refs', color='C2')
    plt.xlabel(f'BJD_TDB - {offset:.1f} [d]')
    plt.ylabel('Differential flux (target/(ref1+ref2))')
    plt.legend(fontsize='small')
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------------
    # 1.5 Selection of the best light curve
    # --------------------------------------------------------------
    print("Plotting sum-of-refs with errorbars and transit flags…")
    oot = (bjd < 2460024.3450) | (bjd > 2460024.4350)
    bjd_med = np.median(bjd)
    med_rel = bjd_med - offset

    plt.figure(figsize=(8, 4), dpi=150)
    plt.errorbar(bjd - offset, diff_all, yerr=err_all, fmt='.', ecolor='gray',
                 elinewidth=0.5, alpha=0.5, label='All refs ±1σ')
    plt.scatter(bjd - offset, diff_all, s=4, c='C2')
    # linea verticale in corrispondenza della mediana
    plt.axvline(med_rel, linestyle='--', label=f'Median ({med_rel:.4f} d)')
    plt.xlabel(f'BJD_TDB - {offset:.1f} [d]')
    plt.ylabel('Differential flux')
    plt.legend(fontsize='small')
    plt.tight_layout()
    plt.show()

    # stampa STD fuori dal transito
    print(f'STD Sum of refs (OOT): {np.std(diff_all[oot]):.7f}')

    std1 = np.std(diff1[oot])
    std2 = np.std(diff2[oot])
    std_all = np.std(diff_all[oot])
    print(f'STD Ref #1: {std1:.7f}')
    print(f'STD Ref #2: {std2:.7f}')
    print(f'STD Sum of refs: {std_all:.7f}')

    # --------------------------------------------------------------
    # 1.6 Using different photometric parameters (aperture = 5 px)
    # --------------------------------------------------------------
    print("Testing aperture = 5 px…")
    aperture_test = 5
    inner_test = 13
    outer_test = 18

    target_ap05 = AperturePhotometry()
    target_ap05.provide_aperture_parameters(inner_test, outer_test,
                                           aperture_test, x_target, y_target)
    target_ap05.aperture_photometry()

    ref1_ap05 = AperturePhotometry()
    ref1_ap05.provide_aperture_parameters(inner_test, outer_test,
                                          aperture_test, x_ref1, y_ref1)
    ref1_ap05.aperture_photometry()

    ref2_ap05 = AperturePhotometry()
    ref2_ap05.provide_aperture_parameters(inner_test, outer_test,
                                          aperture_test, x_ref2, y_ref2)
    ref2_ap05.aperture_photometry()

    diff5_1 = target_ap05.aperture / ref1_ap05.aperture
    diff5_2 = target_ap05.aperture / ref2_ap05.aperture
    sum5 = ref1_ap05.aperture + ref2_ap05.aperture
    diff5_all = target_ap05.aperture / sum5

    bjd_med = np.median(bjd)
    p5_1 = Polynomial.fit(bjd[oot]-bjd_med, diff5_1[oot], deg=1)
    p5_2 = Polynomial.fit(bjd[oot]-bjd_med, diff5_2[oot], deg=1)
    p5_all = Polynomial.fit(bjd[oot]-bjd_med, diff5_all[oot], deg=1)
    norm5_1 = diff5_1 / p5_1(bjd-bjd_med)
    norm5_2 = diff5_2 / p5_2(bjd-bjd_med)
    norm5_all = diff5_all / p5_all(bjd-bjd_med)

    plt.figure(figsize=(8,4), dpi=150)
    plt.scatter(bjd-offset, norm5_1, s=4, label='Ref #1 (5px)')
    plt.scatter(bjd-offset, norm5_2, s=4, label='Ref #2 (5px)')
    plt.scatter(bjd-offset, norm5_all, s=4, label='Both (5px)')
    plt.axvline(bjd_med-offset, linestyle='--')
    plt.ylim(0.95, 1.05)
    plt.xlabel(f'BJD_TDB - {offset:.1f} [d]')
    plt.ylabel('Normalized flux')
    plt.legend(fontsize='small')
    plt.tight_layout()
    plt.show()

    std5_1 = np.std(norm5_1[oot])
    std5_2 = np.std(norm5_2[oot])
    std5_all = np.std(norm5_all[oot])
    print(f'STD 5px Ref #1: {std5_1:.7f}')
    print(f'STD 5px Ref #2: {std5_2:.7f}')
    print(f'STD 5px Both: {std5_all:.7f}')

    # New Section: Save results with exact file names
    # Define variables for saving
    bjd_tdb = bjd
    # Normalize differential flux for target/ref1 and target/(ref1+ref2) using out-of-transit median
    differential_ref01_normalized = diff1 / np.median(diff1[oot])
    differential_ref01_normalized_error = err1 / np.median(diff1[oot])
    differential_allref_normalized = diff_all / np.median(diff_all[oot])
    differential_allref_normalized_error = err_all / np.median(diff_all[oot])
    # Original differential photometry (non-normalized)
    differential_ref01 = diff1
    differential_ref01_error = err1
    differential_allref = diff_all
    differential_allref_error = err_all

    # Save to pickle files using the exact file names provided
    pickle.dump(bjd_tdb, open(f'{taste_dir}/taste_bjdtdb.p','wb'))
    pickle.dump(differential_ref01_normalized, open(f'{taste_dir}/differential_ref01_normalized.p','wb'))
    pickle.dump(differential_ref01_normalized_error, open(f'{taste_dir}/differential_ref01_normalized_error.p','wb'))
    pickle.dump(differential_allref_normalized, open(f'{taste_dir}/differential_allref_normalized.p','wb'))
    pickle.dump(differential_allref_normalized_error, open(f'{taste_dir}/differential_allref_normalized_error.p','wb'))

    pickle.dump(differential_ref01, open(f'{taste_dir}/differential_ref01.p','wb'))
    pickle.dump(differential_ref01_error, open(f'{taste_dir}/differential_ref01_error.p','wb'))

    pickle.dump(differential_allref, open(f'{taste_dir}/differential_allref.p','wb'))
    pickle.dump(differential_allref_error, open(f'{taste_dir}/differential_allref_error.p','wb'))
    
if __name__ == '__main__':
    main()
