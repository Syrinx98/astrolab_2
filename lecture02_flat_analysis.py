"""
2 Flat Analysis
"""
import pickle

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt

# =============================================================================
print("\n\n2.1 Read the data")
print("=============================================================================\n")

"""
2.1 Read the data

We read the list of bias frames in the usual way
"""
taste_dir = 'TASTE_analysis/group05_QATAR-1_20230212'

flat_list = np.genfromtxt(f'{taste_dir}/flat/flat.list', dtype=str)
print(flat_list)

"""
We also need to open the median bias and copy the estimates for the error associated
with the bias and the readout noise obtained when analyzing the bias frames. For this
example, I will use the following values from the lecture01_bias_analysis
"""

median_bias = pickle.load(open(f"{taste_dir}/bias/median_bias.p", "rb"))
bias_std = 1.3  # [e] photoelectrons
readout_noise = 7.4  # [e] photoelectrons
gain = 1.91  # [e/ADU]

flat00_fits = fits.open(f'{taste_dir}/flat/' + flat_list[0])
flat00_data = flat00_fits[0].data * gain

print(
    'CCD Gain         : {0:4.2f} {1:.8s}'.format(flat00_fits[0].header['GAIN'], flat00_fits[0].header.comments['GAIN']))
print('CCD Readout noise: {0:4.2f} {1:.3s}'.format(flat00_fits[0].header['RDNOISE'],
                                                   flat00_fits[0].header.comments['RDNOISE']))
print('Shape of the FITS image from the header : {0:4d} x {1:4d} pixels'.format(flat00_fits[0].header['NAXIS1'],
                                                                                flat00_fits[0].header['NAXIS2']))

# =============================================================================
print("\n\n2.2 Overscan")
print("=============================================================================\n")

"""
2.2 Overscan

It's always a good idea to visually inspect your data. If we let matplotlib automatically choose the range of the 
colorbar, and we plot the average distribution of counts as a function of the column number, 
we can see that there are two strips on the side of the image with a much lower number of counts. 
Each column takes the name of overscan.
"""

fig, ax = plt.subplots(2, 1, figsize=(8, 6))  # Caution, figsize will also influence positions.
im1 = ax[0].imshow(flat00_data, origin='lower')
median_column = np.average(flat00_data, axis=0)
im2 = ax[1].plot(median_column)

# add the colorbar using the figure's method,
# telling which mappable we're talking about and
# which axes object it should be near
cbar = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("e")

ax[0].set_xlabel('X [pixels]')
ax[0].set_ylabel('Y [pixels]')

ax[1].set_xlabel('X [pixels]')
ax[1].set_ylabel('Y [pixels]')
plt.show()

# =============================================================================

print("\n\n2.3 Properly dealing with the entire frame")
print("=============================================================================\n")

"""
2.3 Properly dealing with the entire frame

The overscan regions do not impact our data reduction and science analysis, but their presence is annoying when 
performing statistics on the full frame or displaying it. There are (at least) two ways to deal with this problem:

We can trim the outer columns from all images and save the trimmed frames as new frames.
we can exclude the outer columns from the analysis when computing full-frame statistics and visualization.
The first approach is well-diffused, but you need software that propagates the metadata (e.g., header, comments) 
into the new files. You also have to be sure to apply the same overscan cuts every time you reduce a new frame. 
You may also need extra space to store all the new images. The second approach requires just a bit of attention.

Let's plot our frame again, but first, we compute the minimum and maximum values of our counts, excluding the overscan regions.
"""

vmin = np.amin(flat00_data[:, 12:-12])
vmax = np.amax(flat00_data[:, 12:-12])
print(vmin, vmax)

fig, ax = plt.subplots(2, 1, figsize=(8, 6))  # Caution, figsize will also influence positions.
im1 = ax[0].imshow(flat00_data, origin='lower', vmin=vmin, vmax=vmax)
median_column = np.average(flat00_data, axis=0)
im2 = ax[1].plot(median_column)

# we set the plot limits
ax[1].set_ylim(vmin, vmax)

# add the colorbar using the figure's method,
# telling which mappable we're talking about and
# which axes object it should be near
cbar = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("e")

ax[0].set_xlabel('X [pixels]')
ax[0].set_ylabel('Y [pixels]')

ax[1].set_xlabel('X [pixels]')
ax[1].set_ylabel('Average counts [e]')
plt.show()

# =============================================================================
print("\n\n2.4 Computing the normalization factors")
print("=============================================================================\n")

"""
2.4 Computing the normalization factors

Computation of the median flat follows a similar strategy adopted for the median bias:

we initialize a 3D array called stack, with dimensions equal to the number of images and the two dimensions of a frame
we read the frames one by one, correct for bias, and store them in the stack.
"""

n_images = len(flat_list)
flat00_nparray_dim00, flat00_nparray_dim01 = np.shape(flat00_data)

stack = np.empty([n_images, flat00_nparray_dim00, flat00_nparray_dim01])

for i_flat, flat_name in enumerate(flat_list):
    flat_temp = fits.open(f'{taste_dir}/flat/' + flat_name)
    stack[i_flat, :, :] = flat_temp[0].data * flat_temp[0].header['GAIN'] - median_bias
    flat_temp.close()

"""
Before continuing, we have to remember the goal of flat calibration.

The purpose of flat corrections is to compensate for any non-uniformity in the response of the CCD to light. 
There can be several reasons for the non-uniform response across the detector:

variations in the sensitivity of pixels in the detector.
dust on either the filter or the glass window covering the detector.
vignetting, a dimming in the corners of the image.
anything else in the optical path that affects how much light reaches the sensor.
The fix for the non-uniformity is the same in all cases: take an image with uniform illumination and use that to 
measure the CCD's response.

When illuminating the CCD, we want to reach the highest signal-to-noise ratio for every pixel. 
As the photon noise goes with the square root of the flux, we need to achieve very high counts 
(without reaching saturation). However, dividing the science frames by these very high counts would produce 
unrealistically small photoelectron fluxes. Since we are interested in the relative response of the pixels, i.e., 
how the pixels behave with respect to the others, and not to the absolute efficiency of each pixel, we can express 
the flat correction as the correction value relative to the median response over a selected sample of pixels. e CCD.
We can compute the reference value of each frame by taking the median within a box of 50x50 pixels in the centre 
of each frame. Keep in mind that this is only one of the many possible approaches.
Note: we convert the real number into integers using the function numpy.int16.
Alternatively, we can use the Python built-in function int.
"""

windows_size = 50
# x0, x1, y0, y1 represents the coordinates of the four corners
x0 = np.int16(flat00_nparray_dim01 / 2 - windows_size / 2)
x1 = np.int16(flat00_nparray_dim01 / 2 + windows_size / 2)
y0 = np.int16(flat00_nparray_dim00 / 2 - windows_size / 2)
y1 = np.int16(flat00_nparray_dim00 / 2 + windows_size / 2)

print('Coordinates of the box: x0:{0}, x1:{1}, y0:{2}, y1:{3}'.format(x0, x1, y0, y1))

normalization_factors = np.median(stack[:, y0:y1, x0:x1], axis=(1, 2))
print('Number of normalization factors (must be the same as the number of frames): {0}'.format(
    np.shape(normalization_factors)))
print(normalization_factors)

normalization_factors_std = np.std(stack[:, y0:y1, x0:x1], axis=(1, 2)) / np.sqrt(windows_size ** 2)
print(normalization_factors_std)

plt.figure()
x_frame = np.arange(0, n_images, 1)
plt.scatter(x_frame, normalization_factors)
plt.errorbar(x_frame, normalization_factors, normalization_factors_std, fmt='o', ms=2)
plt.xlabel('Frame number')
plt.ylabel('Average counts [e]')
plt.show()

# =============================================================================
print("\n\n2.5 Flat normalization")
print("=============================================================================\n")

"""
2 Flat normalization

"""

stack_normalized_iter = stack * 0.  # initialization of the output array
for i_flat in range(n_images):
    stack_normalized_iter[i_flat, :, :] = stack[i_flat, :, :] / normalization_factors[i_flat]

print("shape of stack array           : ", np.shape(stack))
print("shape of transposed stack array: ", np.shape(stack.T))

stack_normalized = (stack.T / normalization_factors).T
## First alternative:  stack_normalized = np.divide(stack.T, normalization_factors).T
## Second alternative: stack_normalized = np.multiply(stack.T, 1./normalization_factors).T

print("shape of normalized stack array: ", np.shape(stack_normalized))

print("Maximum absolute difference between the two arrays: {0:2.6e}".format(
    np.max(np.abs(stack_normalized_iter - stack_normalized))))

"""
The value may not be zero due to the computer's precision: Epsilon describes the round-off error for a 
floating-point number with a certain amount of precision. It can be thought of as the smallest number 
that can be added to 1.0 without changing its bits.
"""
print(np.finfo(float).eps)

# =============================================================================
print("\n\n2.6 Median flat")
print("=============================================================================\n")

"""
2.6 Median flat
"""

median_normalized_flat = np.median(stack_normalized, axis=0)

with open(f'{taste_dir}/flat/median_normalized_flat.p', 'wb', buffering=0) as f:
    # noinspection PyTypeChecker
    pickle.dump(median_normalized_flat, f)

with open(f'{taste_dir}/flat/flat_normalized_stack.p', 'wb') as f:
    # noinspection PyTypeChecker
    pickle.dump(stack_normalized, f)

with open(f'{taste_dir}/flat/flat_normalization_factors.p', 'wb') as f:
    # noinspection PyTypeChecker
    pickle.dump(normalization_factors, f)

with open(f'{taste_dir}/flat/flat_stack.p', 'wb') as f:
    # noinspection PyTypeChecker
    pickle.dump(stack, f)

i_image = 0
nmin = np.amin(median_normalized_flat[:, 12:-12])
nmax = np.amax(median_normalized_flat[:, 12:-12])
print(vmin, vmax)

fig, ax = plt.subplots(2, 1, figsize=(8, 6))  # Caution, figsize will also influence positions.
im1 = ax[0].imshow(median_normalized_flat, origin='lower', vmin=nmin, vmax=nmax)
median_column = np.average(median_normalized_flat, axis=0)
im2 = ax[1].plot(median_column)

# we set the plot limits
ax[1].set_ylim(nmin, nmax)

# add the colorbar using the figure's method,
# telling which mappable we're talking about and
# which axes object it should be near
cbar = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("e")

ax[0].set_xlabel('X [pixels]')
ax[0].set_ylabel('Y [pixels]')

ax[1].set_xlabel('X [pixels]')
ax[1].set_ylabel('Average value [normalized]')
plt.show()

# =============================================================================
print("\n\n2.7 Error propagation")
print("=============================================================================\n")

"""
2.7 Error propagation

Regarding the bias frames, the readout noise was the only contributor to the noise budget. Now, we have three contributors:

the readout noise
the error associated with the median bias frame
the photon noise associated with the flux of the lamp
Dark current would contribute as well, but it can be neglected for short exposures and for modern detectors in general.

For 1) and 2), we already have an estimate of the error. For 3), we can assume that during the temporal interval 
covered by our exposure, photons reach the detector with a constant mean rate and independently of the time since the 
last event. (the arrival of one photon is not influenced by the other photons). In other words, 
photos follow a Poisson distribution; as such, the variance (i.e., the expected value of the squared deviation 
from the mean of a random variable.) is equal to the number of events, specifically the number of photoelectrons recorded.

Noting that we have already removed the bias value (which does not contribute to the photon noise), 
the photon noise associated with each measurement of each frame is given by the square root of the stack.
"""

photon_noise = np.sqrt(np.abs(stack))

stack_error = np.sqrt(readout_noise **2 + bias_std**2 +  photon_noise**2)

stack_normalized_error = (stack_error.T/normalization_factors).T

median_normalized_flat_errors = np.sum(stack_normalized_error**2, axis=0) / n_images
print("shape of the median normalized error array: ", np.shape(median_normalized_flat_errors))

with open(f'{taste_dir}/flat/median_normalized_flat_errors.p', 'wb') as f:
    # noinspection PyTypeChecker
    pickle.dump(median_normalized_flat_errors, f)

# =============================================================================
print("\n\n2.8 Some statistics on the flat")
print("=============================================================================\n")

"""
2.8 Some statistics on the flat

One question should arise: why normalize first, and then computing the median? Can I change the order without consequences?

First, let's see what is the distribution of the counts before and after normalization. 
For this example, I'm considering pixels in the range [100:105,250:255] (Python notation). 
The normalized values have been rescaled to the average of the normalization factors to allow a direct comparison.

If the flux of the lamp is constant, the two distributions should be extremely similar - almost identical.
"""

mean_normalization = np.mean(normalization_factors)

plt.figure()
plt.hist(stack[:,40:45,250:255].flatten(), bins=20, alpha=0.5, label='Before norm.')
plt.hist(stack_normalized[:,40:45,250:255].flatten()*mean_normalization, bins=20, alpha=0.5, label='After norm.')
plt.xlabel('Counts [e]')
plt.ylabel('#')
plt.legend()
plt.show()

"""
There is a clear difference between the two distributions (can you quanitfy it)? 
The reason appears clear if we plot the distribution of the normalization factors, 
and we compare it with the theoretical distribution that they should have if the variations 
was due to photon noise alone.
"""

sigma_mean_normalization = np.sqrt(mean_normalization)
x = np.arange(np.amin(normalization_factors), np.amax(normalization_factors), 10)
y = 1./(sigma_mean_normalization * np.sqrt(2 * np.pi)) * \
               np.exp( - (x - mean_normalization)**2 / (2 * sigma_mean_normalization**2) )

plt.figure()
plt.hist(normalization_factors,alpha=0.5, density=True, label='Normalization factors')
plt.plot(x,y)
plt.xlabel('Counts [e]')
plt.ylabel('Probability density')
plt.legend()
plt.show()

"""
t appears clear that the variation in the illumination by the flat lamp 
is not consistent with photon noise alone. After technical investigation, 
a fluctuations in the voltage of the lamp has been discovered.

For this reason, we performed the normalization, i.e., removing the variation of 
illumination with time, before computing the median.
"""