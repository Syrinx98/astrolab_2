"""
6 Classes: Aperture Photometry Class with JD to BJD_TDB conversion and additional outputs

In this version, we add attributes needed for plotting telescope drift and FWHM:
- x_refined and y_refined arrays are essentially the final centroid positions of the star for each image.
  We have already stored these in self.x_position and self.y_position. For convenience and consistency,
  we can either rename them or just use self.x_position and self.y_position where needed.
  If the external code expects target_ap08.x_refined and target_ap08.y_refined, we can set them as aliases
  after finishing aperture_photometry.

- x_fwhm and y_fwhm must be computed from the star's flux distribution. We replicate the approach
  used in previous examples: after measuring the star position and subtracting the sky, we measure
  the Full Width at Half Maximum (FWHM) along the x and y axes by analyzing the cumulative flux profile.

We implement a helper method determine_FWHM_axis to compute the FWHM along a given axis, and a step
inside aperture_photometry to compute self.x_fwhm and self.y_fwhm for each image.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle
from astropy.io import fits
from astropy import coordinates as coord, units as u
from astropy.time import Time

class AperturePhotometry:
    taste_dir = 'TASTE_analysis/group05_QATAR-1_20230212'

    def __init__(self):
        # Initialization: load constants and calibration frames
        self.data_path = self.taste_dir + "/"

        self.readout_noise = 7.4  # [e] photoelectrons
        self.gain = 1.91  # [e/ADU]
        self.bias_std = 1.3  # [e] photoelectrons

        # Load median bias and errors
        self.median_bias = pickle.load(open(self.data_path + 'bias/median_bias.p', 'rb'))
        self.median_bias_errors = pickle.load(open(self.data_path + 'bias/median_bias_error.p', 'rb'))

        # Load median flat and errors
        self.median_normalized_flat = pickle.load(open(self.data_path + 'flat/median_normalized_flat.p', 'rb'))
        self.median_normalized_flat_errors = pickle.load(open(self.data_path + 'flat/median_normalized_flat_errors.p', 'rb'))

        # Science files
        self.science_path = self.data_path + 'science/'
        self.science_list = np.genfromtxt(self.science_path + 'science.list', dtype=str)
        self.science_size = len(self.science_list)

        # Create meshgrid
        ylen, xlen = np.shape(self.median_bias)
        X_axis = np.arange(0, xlen, 1)
        Y_axis = np.arange(0, ylen, 1)
        self.X, self.Y = np.meshgrid(X_axis, Y_axis)
        self.X_axis = X_axis
        self.Y_axis = Y_axis

        # Set the target coordinates and observatory location for BJD_TDB conversion
        self.target = coord.SkyCoord("20:13:31.61","+65:09:43.49", unit=(u.hourangle, u.deg), frame='icrs')
        self.observatory_location = ('45.8472d', '11.569d')

    def provide_aperture_parameters(self, sky_inner_radius, sky_outer_radius, aperture_radius, x_initial, y_initial):
        self.sky_inner_radius = sky_inner_radius
        self.sky_outer_radius = sky_outer_radius
        self.aperture_radius = aperture_radius
        self.x_initial = x_initial
        self.y_initial = y_initial

    def correct_science_frame(self, science_frame):
        # Correct science frame for bias and flat
        science_debiased = science_frame - self.median_bias
        science_corrected = science_debiased / self.median_normalized_flat

        # Error computation
        science_debiased_errors = np.sqrt(self.readout_noise ** 2 + self.bias_std ** 2 + science_debiased)

        valid = (science_debiased != 0) & (self.median_normalized_flat != 0)
        science_corrected_errors = np.zeros_like(science_corrected)
        science_corrected_errors[valid] = science_corrected[valid] * np.sqrt(
            (science_debiased_errors[valid] / science_debiased[valid]) ** 2 +
            (self.median_normalized_flat_errors[valid] / self.median_normalized_flat[valid]) ** 2
        )
        # For invalid pixels:
        science_corrected_errors[~valid] = 0.0

        return science_corrected, science_corrected_errors

    def compute_centroid(self, science_frame, x_target_initial, y_target_initial, maximum_number_of_iterations=20):
        for i_iter in range(0, maximum_number_of_iterations):
            if i_iter == 0:
                x_target_previous = x_target_initial
                y_target_previous = y_target_initial
            else:
                x_target_previous = x_target_refined
                y_target_previous = y_target_refined

            target_distance = np.sqrt((self.X - x_target_previous) ** 2 + (self.Y - y_target_previous) ** 2)
            annulus_sel = (target_distance < self.sky_inner_radius)

            weighted_X = np.sum(science_frame[annulus_sel] * self.X[annulus_sel])
            weighted_Y = np.sum(science_frame[annulus_sel] * self.Y[annulus_sel])
            total_flux = np.sum(science_frame[annulus_sel])

            x_target_refined = weighted_X / total_flux
            y_target_refined = weighted_Y / total_flux

            percent_variance_x = (x_target_refined - x_target_previous) / x_target_previous * 100.
            percent_variance_y = (y_target_refined - y_target_previous) / y_target_previous * 100.

            if np.abs(percent_variance_x) < 0.1 and np.abs(percent_variance_y) < 0.1:
                break

        return x_target_refined, y_target_refined

    def compute_sky_background(self, science_frame, science_frame_errors, x_pos, y_pos):
        target_distance = np.sqrt((self.X - x_pos) ** 2 + (self.Y - y_pos) ** 2)
        annulus_selection = (target_distance > self.sky_inner_radius) & (target_distance <= self.sky_outer_radius)

        sky_flux_median = np.median(science_frame[annulus_selection])
        N = np.sum(annulus_selection)
        sky_flux_error = np.sqrt(np.sum(science_frame_errors[annulus_selection] ** 2)) / N

        return sky_flux_median, sky_flux_error

    def determine_FWHM_axis(self, reference_axis, normalized_cumulative_distribution):
        """
        Determine the FWHM along one axis using the normalized cumulative distribution.

        We approximate the -1 sigma (0.15865) and +1 sigma (0.84135) positions from a Gaussian
        and then convert to FWHM using FWHM_factor = 2 * sqrt(2 * ln(2)) ~ 2.35482.

        We fit a small polynomial around the points to get more precise estimates.
        """
        # Find closest points to NCD=0.15865 and 0.84135
        # Handle cases where these values may not be perfectly found
        NCD_left_val = 0.15865
        NCD_right_val = 0.84135

        NCD_index_left = np.argmin(np.abs(normalized_cumulative_distribution - NCD_left_val))
        NCD_index_right = np.argmin(np.abs(normalized_cumulative_distribution - NCD_right_val))

        # Fit a small polynomial around these indices to interpolate more accurately
        from numpy.polynomial import Polynomial

        # Ensure indices are valid
        left_range = slice(max(NCD_index_left-1,0), min(NCD_index_left+2, len(reference_axis)))
        right_range = slice(max(NCD_index_right-1,0), min(NCD_index_right+2,len(reference_axis)))

        # Fit around left point
        p_fitted_left = Polynomial.fit(normalized_cumulative_distribution[left_range],
                                       reference_axis[left_range],
                                       deg=2)
        pixel_left = p_fitted_left(NCD_left_val)

        # Fit around right point
        p_fitted_right = Polynomial.fit(normalized_cumulative_distribution[right_range],
                                        reference_axis[right_range],
                                        deg=2)
        pixel_right = p_fitted_right(NCD_right_val)

        FWHM_factor = 2 * np.sqrt(2 * np.log(2)) # ~2.35482
        # pixel_right - pixel_left is ~2 sigma width, so:
        FWHM = (pixel_right - pixel_left)/2. * FWHM_factor

        return FWHM

    def compute_fwhm(self, science_frame, x_pos, y_pos, radius):
        """
        Compute FWHM along X and Y directions for the star located at x_pos, y_pos.
        We select pixels within radius and compute normalized cumulative distributions along x and y.
        """
        target_distance = np.sqrt((self.X - x_pos)**2 + (self.Y - y_pos)**2)
        sel = (target_distance < radius)

        total_flux = np.nansum(science_frame * sel)
        # flux along each axis
        flux_x = np.nansum(science_frame * sel, axis=0)
        flux_y = np.nansum(science_frame * sel, axis=1)

        cumulative_sum_x = np.cumsum(flux_x) / total_flux
        cumulative_sum_y = np.cumsum(flux_y) / total_flux

        FWHM_x = self.determine_FWHM_axis(self.X_axis, cumulative_sum_x)
        FWHM_y = self.determine_FWHM_axis(self.Y_axis, cumulative_sum_y)

        return FWHM_x, FWHM_y

    def aperture_photometry(self):
        self.airmass = np.empty(self.science_size)
        self.exptime = np.empty(self.science_size)
        self.julian_date = np.empty(self.science_size)

        self.aperture = np.empty(self.science_size)
        self.aperture_errors = np.empty(self.science_size)
        self.sky_background = np.empty(self.science_size)
        self.sky_background_errors = np.empty(self.science_size)

        self.x_position = np.empty(self.science_size)
        self.y_position = np.empty(self.science_size)

        # We also create arrays for FWHM measurements
        self.x_fwhm = np.empty(self.science_size)
        self.y_fwhm = np.empty(self.science_size)

        x_ref_init = self.x_initial
        y_ref_init = self.y_initial

        for ii_science, science_name in enumerate(self.science_list):
            science_fits = fits.open(self.science_path + science_name)
            self.airmass[ii_science] = science_fits[0].header['AIRMASS']
            self.exptime[ii_science] = science_fits[0].header['EXPTIME']
            self.julian_date[ii_science] = science_fits[0].header['JD']

            science_data = science_fits[0].data * self.gain
            science_fits.close()

            # Correct frame
            science_corrected, science_corrected_errors = self.correct_science_frame(science_data)

            # Compute centroid before sky subtraction
            x_refined, y_refined = self.compute_centroid(science_corrected, x_ref_init, y_ref_init)

            # Compute sky background
            sky_median, sky_error = self.compute_sky_background(science_corrected, science_corrected_errors, x_refined, y_refined)
            self.sky_background[ii_science] = sky_median
            self.sky_background_errors[ii_science] = sky_error

            # Subtract sky
            science_sky_corrected = science_corrected - self.sky_background[ii_science]
            science_sky_corrected_errors = np.sqrt(science_corrected_errors ** 2 + self.sky_background_errors[ii_science] ** 2)

            # Recompute centroid after sky subtraction
            x_refined, y_refined = self.compute_centroid(science_sky_corrected, x_refined, y_refined)

            # Aperture photometry
            target_distance = np.sqrt((self.X - x_refined) ** 2 + (self.Y - y_refined) ** 2)
            aperture_selection = (target_distance < self.aperture_radius)
            self.aperture[ii_science] = np.sum(science_sky_corrected[aperture_selection])
            self.aperture_errors[ii_science] = np.sqrt(np.sum((science_sky_corrected_errors[aperture_selection]) ** 2))

            self.x_position[ii_science] = x_refined
            self.y_position[ii_science] = y_refined

            # Compute FWHM after we have the final centroid and sky-subtracted frame
            # We use self.sky_inner_radius as radius for FWHM measurement, as it should encompass the star flux.
            fwhm_x, fwhm_y = self.compute_fwhm(science_sky_corrected, x_refined, y_refined, radius=self.sky_inner_radius)
            self.x_fwhm[ii_science] = fwhm_x
            self.y_fwhm[ii_science] = fwhm_y

            # Update initial guess for next frame
            x_ref_init = x_refined
            y_ref_init = y_refined

        # Convert JD to BJD_TDB
        jd_mid = self.julian_date + self.exptime/86400./2.
        tm = Time(jd_mid, format='jd', scale='utc', location=self.observatory_location)
        ltt_bary = tm.light_travel_time(self.target, ephemeris='jpl')
        self.bjd_tdb = tm.tdb + ltt_bary

        # For compatibility with the code that uses x_refined, y_refined:
        self.x_refined = self.x_position
        self.y_refined = self.y_position

# Example usage:
from time import time
t0 = time()
target_star = AperturePhotometry()
target_star.provide_aperture_parameters(13, 18, 8, 415, 73)
target_star.aperture_photometry()
t1 = time()
print('elapsed_time=', t1-t0)

plt.figure()
plt.scatter(target_star.bjd_tdb.value, target_star.aperture, s=2)
plt.xlabel("BJD_TDB")
plt.ylabel("Aperture flux")
plt.show()
