#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6 Classes: Aperture Photometry Class with JD → BJD_TDB conversion and additional outputs,
plus detrending to remove linear background slope (“inclinazione”).
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from astropy.io import fits
from astropy import coordinates as coord, units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation

class AperturePhotometry:
    taste_dir = 'TASTE_analysis/group05_QATAR-1_20230212'

    def __init__(self):
        # Initialization: load constants and calibration frames
        self.data_path = self.taste_dir + "/"
        self.readout_noise = 7.4  # [e-] photoelectrons
        self.gain = 1.91         # [e-/ADU]
        self.bias_std = 1.3      # [e-] photoelectrons

        # Load median bias and errors
        print("Loading calibration frames...")
        self.median_bias = pickle.load(open(self.data_path + 'bias/median_bias.p', 'rb'))
        self.median_bias_errors = pickle.load(open(self.data_path + 'bias/median_bias_error.p', 'rb'))

        # Load median flat and errors
        self.median_normalized_flat = pickle.load(open(self.data_path + 'flat/median_normalized_flat.p', 'rb'))
        self.median_normalized_flat_errors = pickle.load(open(self.data_path + 'flat/median_normalized_flat_errors.p', 'rb'))
        print("Calibration frames loaded.\n")

        # Science files
        self.science_path = self.data_path + 'science/'
        self.science_list = np.genfromtxt(self.science_path + 'science.list', dtype=str)
        self.science_size = len(self.science_list)
        print(f"Found {self.science_size} science frames.\n")

        # Create meshgrid
        ylen, xlen = self.median_bias.shape
        X_axis = np.arange(xlen)
        Y_axis = np.arange(ylen)
        self.X, self.Y = np.meshgrid(X_axis, Y_axis)
        self.X_axis = X_axis
        self.Y_axis = Y_axis

        # Target coords & observatory location for BJD_TDB conversion
        self.target = coord.SkyCoord("20:13:31.61", "+65:09:43.49",
                                     unit=(u.hourangle, u.deg), frame='icrs')
        self.observatory_location = EarthLocation(lat=45.8472*u.deg,
                                                  lon=11.569*u.deg)

    def provide_aperture_parameters(self, sky_inner_radius, sky_outer_radius,
                                    aperture_radius, x_initial, y_initial):
        self.sky_inner_radius = sky_inner_radius
        self.sky_outer_radius = sky_outer_radius
        self.aperture_radius = aperture_radius
        self.x_initial = x_initial
        self.y_initial = y_initial
        print(f"Aperture params → sky_inner={sky_inner_radius}, sky_outer={sky_outer_radius}, aperture={aperture_radius}")
        print(f"Initial guess → x0={x_initial}, y0={y_initial}\n")

    def correct_science_frame(self, science_frame):
        # Bias subtraction & flat-field correction with error propagation
        with np.errstate(divide='ignore', invalid='ignore'):
            debiased = science_frame - self.median_bias
            corrected = debiased / self.median_normalized_flat

            debias_err = np.sqrt(self.readout_noise**2 +
                                 self.bias_std**2 +
                                 np.abs(debiased))

            valid = (debiased != 0) & (self.median_normalized_flat != 0)
            corr_err = np.zeros_like(corrected)
            corr_err[valid] = corrected[valid] * np.sqrt(
                (debias_err[valid] / np.abs(debiased[valid]))**2 +
                (self.median_normalized_flat_errors[valid] / self.median_normalized_flat[valid])**2
            )

            corrected[~valid] = 0.0

        return corrected, corr_err

    def compute_centroid(self, frame, x0, y0, max_iter=20, tol=0.1):
        """
        Iteratively refine centroid within sky_inner_radius.
        Returns last valid (x, y), falling back to initial if no flux.
        """
        x_prev, y_prev = x0, y0

        for _ in range(max_iter):
            r = np.hypot(self.X - x_prev, self.Y - y_prev)
            sel = (r < self.sky_inner_radius)
            flux = frame[sel]
            total = np.sum(flux)
            if total <= 0:
                break
            x_new = np.sum(flux * self.X[sel]) / total
            y_new = np.sum(flux * self.Y[sel]) / total

            dx = abs((x_new - x_prev) / x_prev)*100 if x_prev else np.inf
            dy = abs((y_new - y_prev) / y_prev)*100 if y_prev else np.inf
            x_prev, y_prev = x_new, y_new
            if dx < tol and dy < tol:
                break

        return x_prev, y_prev

    def compute_sky_background(self, frame, frame_err, x_cen, y_cen):
        r = np.hypot(self.X - x_cen, self.Y - y_cen)
        ann = (r > self.sky_inner_radius) & (r <= self.sky_outer_radius)
        vals = frame[ann]
        errs = frame_err[ann]
        med = np.median(vals)
        Npix = np.count_nonzero(ann)
        err = np.sqrt(np.sum(errs**2)) / Npix if Npix else np.nan
        return med, err

    def determine_FWHM_axis(self, coords, cdf):
        # Linear interpolation at ±1σ points
        left, right = 0.15865, 0.84135
        x_l = np.interp(left,  cdf, coords)
        x_r = np.interp(right, cdf, coords)
        return np.sqrt(2*np.log(2)) * (x_r - x_l)

    def compute_fwhm(self, frame, x_cen, y_cen):
        r = np.hypot(self.X - x_cen, self.Y - y_cen)
        sel = (r < self.sky_inner_radius)
        total = np.sum(frame[sel])
        if total <= 0:
            return np.nan, np.nan
        fx = np.sum(frame*sel, axis=0)
        fy = np.sum(frame*sel, axis=1)
        cdf_x = np.cumsum(fx) / total
        cdf_y = np.cumsum(fy) / total
        return (self.determine_FWHM_axis(self.X_axis, cdf_x),
                self.determine_FWHM_axis(self.Y_axis, cdf_y))

    def aperture_photometry(self):
        N = self.science_size
        # prepare arrays
        self.airmass = np.empty(N)
        self.exptime = np.empty(N)
        self.jd = np.empty(N)
        self.aperture = np.empty(N)
        self.aperture_errors = np.empty(N)
        self.sky_background = np.empty(N)
        self.sky_background_errors = np.empty(N)
        self.x_position = np.empty(N)
        self.y_position = np.empty(N)
        self.x_fwhm = np.empty(N)
        self.y_fwhm = np.empty(N)

        x_ref, y_ref = self.x_initial, self.y_initial
        print("Starting aperture photometry...\n")
        for i, fname in enumerate(self.science_list, 1):
            print(f"[{i}/{N}] {fname}")
            with fits.open(self.science_path + fname) as hdul:
                hdr  = hdul[0].header
                data = hdul[0].data.astype(float) * self.gain

            self.airmass[i-1] = hdr.get('AIRMASS', np.nan)
            self.exptime[i-1] = hdr.get('EXPTIME', np.nan)
            self.jd[i-1]      = hdr.get('JD',     np.nan)

            corr, corr_err = self.correct_science_frame(data)
            x_ref, y_ref = self.compute_centroid(corr, x_ref, y_ref)
            sky_m, sky_e = self.compute_sky_background(corr, corr_err, x_ref, y_ref)
            self.sky_background[i-1], self.sky_background_errors[i-1] = sky_m, sky_e

            corr_sky = corr - sky_m
            corr_sky_err = np.sqrt(corr_err**2 + sky_e**2)
            x_ref, y_ref = self.compute_centroid(corr_sky, x_ref, y_ref)
            self.x_position[i-1], self.y_position[i-1] = x_ref, y_ref

            r = np.hypot(self.X - x_ref, self.Y - y_ref)
            sel = (r < self.aperture_radius)
            vals = corr_sky[sel]; errs = corr_sky_err[sel]
            self.aperture[i-1] = np.sum(vals)
            self.aperture_errors[i-1] = np.sqrt(np.sum(errs**2))

            try:
                fx, fy = self.compute_fwhm(corr_sky, x_ref, y_ref)
            except Exception:
                fx, fy = np.nan, np.nan
            self.x_fwhm[i-1], self.y_fwhm[i-1] = fx, fy

        print("\nConverting JD → BJD_TDB...")
        jd_mid = self.jd + (self.exptime/86400.)/2.
        t = Time(jd_mid, format='jd', scale='utc', location=self.observatory_location)
        ltt = t.light_travel_time(self.target, ephemeris='jpl')
        self.bjd_tdb = t.tdb + ltt
        print("Done.\n")


if __name__ == "__main__":
    from time import time

    # ---- Target star ----
    t0 = time()
    target_star = AperturePhotometry()
    target_star.provide_aperture_parameters(13, 18, 8, 415, 73)
    target_star.aperture_photometry()
    print(f"Elapsed time: {time()-t0:.1f} s\n")

    # Raw flux vs BJD_TDB
    bjd = target_star.bjd_tdb.value
    flux = target_star.aperture
    plt.figure(figsize=(10,6))
    plt.scatter(bjd, flux, s=5, alpha=0.6)
    plt.xlabel("BJD_TDB"); plt.ylabel("Aperture flux")
    plt.title("Raw Aperture Photometry"); plt.grid(True); plt.tight_layout()
    plt.show()

    # ---- Detrending: remove linear slope via time fit ----
    # 1) Normalize around median
    rel_flux = flux / np.median(flux)
    # 2) Fit and remove linear trend in time
    coeff = np.polyfit(bjd, rel_flux, 1)
    baseline = np.polyval(coeff, bjd)
    detr_flux = rel_flux / baseline

    # 3) Plot detrended curve
    plt.figure(figsize=(10,6))
    plt.scatter(bjd, detr_flux, s=5, alpha=0.6)
    plt.axhline(1.0, color='k', linestyle='--')
    plt.xlabel("BJD_TDB"); plt.ylabel("Normalized & Detrended Flux")
    plt.title("Detrended Light Curve"); plt.grid(True); plt.tight_layout()
    plt.show()
