"""
3 Classes: Aperture Photometry Class
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle
from astropy.io import fits



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
        # In the original instructions, median_bias_errors was given in another file
        # We'll assume it exists and is consistent:
        self.median_bias_errors = pickle.load(open(self.data_path + 'bias/median_bias_error.p', 'rb'))

        # Load median flat and errors
        self.median_normalized_flat = pickle.load(open(self.data_path + 'flat/median_normalized_flat.p', 'rb'))
        self.median_normalized_flat_errors = pickle.load(
            open(self.data_path + 'flat/median_normalized_flat_errors.p', 'rb'))

        # Science files
        self.science_path = self.data_path + 'science/'
        self.science_list = np.genfromtxt(self.science_path + 'science.list', dtype=str)
        self.science_size = len(self.science_list)

        # Create meshgrid
        ylen, xlen = np.shape(self.median_bias)
        X_axis = np.arange(0, xlen, 1)
        Y_axis = np.arange(0, ylen, 1)
        self.X, self.Y = np.meshgrid(X_axis, Y_axis)

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
        # science_debiased_errors = sqrt(readout_noise^2 + bias_std^2 + science_debiased)
        science_debiased_errors = np.sqrt(self.readout_noise ** 2 + self.bias_std ** 2 + science_debiased)

        # science_corrected_errors = science_corrected * sqrt((science_debiased_errors/science_debiased)^2 +
        #                                                     (median_normalized_flat_errors/median_normalized_flat)^2)
        # Take care of divisions by zero. Where science_debiased=0, handle it safely:
        valid = (science_debiased != 0) & (self.median_normalized_flat != 0)
        science_corrected_errors = np.zeros_like(science_corrected)
        # Compute only where valid
        science_corrected_errors[valid] = science_corrected[valid] * np.sqrt(
            (science_debiased_errors[valid] / science_debiased[valid]) ** 2 +
            (self.median_normalized_flat_errors[valid] / self.median_normalized_flat[valid]) ** 2
        )
        # For invalid pixels (where division by zero might occur), set errors to 0 or NaN:
        # Let's set them to 0 safely
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

            # 2D array with the distance of each pixel from the target star
            target_distance = np.sqrt((self.X - x_target_previous) ** 2 + (self.Y - y_target_previous) ** 2)

            # Selection of the pixels within the inner radius (to compute centroid)
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
        # compute sky background in the annulus
        target_distance = np.sqrt((self.X - x_pos) ** 2 + (self.Y - y_pos) ** 2)
        annulus_selection = (target_distance > self.sky_inner_radius) & (target_distance <= self.sky_outer_radius)

        # Using median as sky level
        sky_flux_median = np.median(science_frame[annulus_selection])

        # Compute sky error:
        # Let's approximate sky error as error on the mean for simplicity:
        # We'll use the median as estimate, but for error we can do:
        # error ~ sqrt(sum(errors^2))/N for the pixels in the annulus
        N = np.sum(annulus_selection)
        sky_flux_error = np.sqrt(np.sum(science_frame_errors[annulus_selection] ** 2)) / N

        return sky_flux_median, sky_flux_error

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

        # Use the initial guess provided by user
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

            # Compute centroid once before sky subtraction (optional step)
            x_refined, y_refined = self.compute_centroid(science_corrected, x_ref_init, y_ref_init)

            # Compute sky background on corrected frame
            sky_median, sky_error = self.compute_sky_background(science_corrected, science_corrected_errors, x_refined,
                                                                y_refined)
            self.sky_background[ii_science] = sky_median
            self.sky_background_errors[ii_science] = sky_error

            # Subtract sky
            science_sky_corrected = science_corrected - self.sky_background[ii_science]
            # Error propagation: sqrt(err_frame^2 + sky_err^2)
            science_sky_corrected_errors = np.sqrt(
                science_corrected_errors ** 2 + self.sky_background_errors[ii_science] ** 2)

            # Recompute centroid after sky subtraction (better accuracy)
            x_refined, y_refined = self.compute_centroid(science_sky_corrected, x_refined, y_refined)

            # Aperture photometry
            target_distance = np.sqrt((self.X - x_refined) ** 2 + (self.Y - y_refined) ** 2)
            aperture_selection = (target_distance < self.aperture_radius)
            self.aperture[ii_science] = np.sum(science_sky_corrected[aperture_selection])
            self.aperture_errors[ii_science] = np.sqrt(np.sum((science_sky_corrected_errors[aperture_selection]) ** 2))

            self.x_position[ii_science] = x_refined
            self.y_position[ii_science] = y_refined

            # Update initial guess for next frame iteration:
            x_ref_init = x_refined
            y_ref_init = y_refined

# Example usage:
from time import time
t0 = time()
target_star = AperturePhotometry()
target_star.provide_aperture_parameters(13, 18, 8, 415, 73)
target_star.aperture_photometry()
t1 = time()
print('elapsed_time=', t1-t0)

# You can then plot:
plt.figure()
plt.scatter(target_star.julian_date, target_star.aperture, s=2)
plt.show()
