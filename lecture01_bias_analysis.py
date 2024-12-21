"""
1 Bias analysis
"""
import pickle

import numpy as np

import os
import glob

from astropy.io import fits
from matplotlib import pyplot as plt

# =============================================================================
print("\n\n1.1 Download the dataset and move it to the appropriate location")
print("=============================================================================\n")
print("The dataset is available at the following link:")
print("https://drive.google.com/drive/folders/17FBWwLrFVDm5L7hEC9ZELETUy-sMqhjF")

"""
1.1 Download the dataset and move it to the appropriate location
https://drive.google.com/drive/folders/17FBWwLrFVDm5L7hEC9ZELETUy-sMqhjF
"""

# =============================================================================
print("\n\n1.2 Write the FITS list files for each subdirectory")
print("=============================================================================\n")

"""
1.2 Write the FITS list files for each subdirectory
We have to have .list files for each subdirectory in the dataset.
"""

# take each file from
# TASTE_analysis\group05_QATAR-1_20230212\bias
# TASTE_analysis\group05_QATAR-1_20230212\flat
# TASTE_analysis\group05_QATAR-1_20230212\science
# and write them to a .list file inside the respective directory


taste_dir = 'TASTE_analysis/group05_QATAR-1_20230212'

# get the directories bias, flat and science separately
dirs = ['bias', 'flat', 'science']

# loop over the directories
for d in dirs:
    # get the files in the directory
    files = glob.glob(f'{taste_dir}/{d}/*.fits')
    # write the filenames to a .list file
    with open(f'{taste_dir}/{d}/{d}.list', 'w') as f:
        for file in files:
            filename = os.path.basename(file)
            f.write(f'{filename}\n')

# print bias.list, as a list of strings
bias_list = np.genfromtxt(f'{taste_dir}/bias/bias.list', dtype=str)
print(bias_list)

# =============================================================================
print("\n\n1.3 Extract information from the FITS file")
print("=============================================================================\n")

"""
1.3 Extract information from the FITS file

We use the standard astropy.io.fits routines to 1) open a single FITS file and save it into
a variable named bias00_fits 2) store the Primary HDU of the fits file into a variable named
bias00_hdu 3) print the header of the Primary HDU using the corresponding astropy.io.fits
method
"""

bias00_fits = fits.open(f'{taste_dir}/bias/{bias_list[0]}')
bias00_hdu = bias00_fits[0]
# pretty print the header
for keyword, value in bias00_hdu.header.items():
    print(f'{keyword}: {value}')

# examples of extracting information from the header
bias00_time = bias00_hdu.header['JD']
bias00_airmass = bias00_hdu.header['AIRMASS']
bias00_gain = bias00_hdu.header['GAIN']
bias00_gain_comment = bias00_hdu.header.comments['GAIN']
bias00_readout_noise = bias00_hdu.header['RDNOISE']
bias00_ron_comment = bias00_hdu.header.comments['RDNOISE']
print('Julian date : {0:12.6f} JD'.format(bias00_time))
print('CCD Gain : {0:4.6f} {1:.8s}'.format(bias00_gain,bias00_gain_comment))
print('CCD Readout noise: {0:4.6f} {1:.3s}'.format(bias00_readout_noise,bias00_ron_comment))


# =============================================================================
print("\n\n1.4 Understanding the shape of an image/array in Python")
print("=============================================================================\n")

"""
1.4 Understanding the shape of an image/array in Python

To compute the median of all our frames, we need to open and save them somewhere in the Random
Access Memory of the computer. That means that we have to create a buffer or stack with enough
space (=correct dimensions) to keep all the images.
But how are the images saved into the memory of the computer? The header of the FITS files can
tell us the shape of the picture, i.e., the NAXIS1 and NAXIS2, which we may expect to be the length
of the horizontal axis (number of columns) and the length of the vertical axis (number of row),
respectively.
"""

bias00_naxis1 = bias00_hdu.header['NAXIS1']
bias00_naxis2 = bias00_hdu.header['NAXIS2']
print('Shape of the FITS image from the header : {0:4d} x {1:4d}'.format(bias00_naxis1, bias00_naxis2))
bias00_data = bias00_hdu.data * bias00_gain
bias00_nparray_dim00, bias00_nparray_dim01 = np.shape(bias00_data)
print('Shape of the NumPy array extracted by astropy: {0:4d} x {1:4d}'.format(bias00_nparray_dim00, bias00_nparray_dim01))
print('Our image is saved as a ', type(bias00_data))

# =============================================================================
print("\n\n1.5 Save the bias frames into a 3D array")
print("=============================================================================\n")

"""
1.5 Save the bias frames into a 3D array

To compute the median for each pixel using all our frames, we need to open and save all the images
in the RAM.
We first check how many images we have by computing the length of the bias list - commented
lines will not be included in the computation of the length. We then retrieve the correct size of our
frame according to how Python saved it in the computer’s memory. Finally, we make our empty
array stack where we are going to store all the images
"""

n_images = len(bias_list)
bias00_nparray_dim00, bias00_nparray_dim01 = np.shape(bias00_data)
stack = np.empty([n_images, bias00_nparray_dim00, bias00_nparray_dim01])

for i_bias, bias_name in enumerate(bias_list):
    bias_temp = fits.open(f'{taste_dir}/bias/{bias_name}')
    bias_data = bias_temp[0].data * bias00_gain
    stack[i_bias] = bias_data

median_bias = np.median(stack, axis=0)
np.shape(median_bias)

print('Shape of the median bias frame: ', np.shape(median_bias))

# =============================================================================
print("\n\n1.6 Plotting a single bias and a median bias")
print("=============================================================================\n")

"""
1.6 Plotting a single bias and a median bias

We use the utility matplotlib.pyplot.subplots to include two plots in the same image. The first
two values identify the number of rows andcolumns of the subplot grid (in the example, 2 plots one
on top of the other, spanning the entire horizontal range). We pass the argument figsize=(8,6)
to define the width and height of the plot, in inches. In geenral, we can pass every argument
accepted by matplotlib.pyplot.figure.
"""

# ATTENTION, you may need to install msvc-runtime -> pip install msvc-runtime

fig, ax = plt.subplots(2,1, figsize=(8,6)) # Caution, figsize will also␣influence positions.
im1 = ax[0].imshow(bias00_data, vmin = 2770, vmax =2790, origin='lower')
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
print("\n\n1.7 Statistical analysis of the bias")
print("=============================================================================\n")

"""
1.7 Statistical analysis of the bias

A bias frame comprises two primary components: the offset introduced by the electronics, and
supposedly constant across the frame, and the readout noise. We can then use the bias to extract
some information regarding the readout noise.
First of all, let’s verify if the bias is indeed constant across the frame. Visually, we can highlight
any variation in the bias by restricting the range of the colorbar. Analytically, we can compute the
average across each column and plot the results as a function of the column number. We chose this
way because the plot highlights a more substantial variation across the horizontal direction.

"""

fig, ax = plt.subplots(2,1, figsize=(8,6))
im1 = ax[0].imshow(median_bias, vmin = 2777, vmax =2781, origin='lower')
median_column = np.average(median_bias, axis=0)
im2 = ax[1].plot(median_column)
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

starting_column= 300
ending_column = 350
print('Readout noise : {0:4.6f} e'.format(bias00_readout_noise))
print('STD single frame : {0:4.6f} e'.format(np.std(bias00_data[:,starting_column:ending_column])))
"""
The values above refer to the readout noise of a single exposure. The median bias is the combination
of 30 individual frames, so the associated error will be smaller than the one coming with a single frame.
Suppose we approximate the median with the average function. In that case, we can use the standard equation for error 
propagation to compute the associated error to the median bias, assuming
that all the pixels from each bias are affected by the same error. The resulting value will change
slightly if we use the readout noise as an estimate of the error associated with each image (case 1)
or if we use the standard deviation of a bias frame (case 2)
"""
expected_noise_medianbias = bias00_readout_noise/np.sqrt(n_images)
print('1) Expected noise of median bias : {0:4.6f} e'.format(expected_noise_medianbias))
expected_std_medianbias = np.std(bias00_data[:,starting_column:ending_column])/np.sqrt(n_images)
print('2) Expected STD of median bias : {0:4.6f} e'.format(expected_std_medianbias))


"""
Alternatively, we can compute the error associated with the median bias from the data. We can
use the standard deviation of the median bias, selecting the same range in columns as done before
(case 3), or we can compute the standard deviation of each pixel across all the frames, divide
by the square root of the number of images, and finally calculate the median of all these values.
We perform the last step to have one value for the error associated with the median bias, but in
principle, we could keep the entire frame.
"""

measured_std_medianbias = np.std(median_bias[:,starting_column:ending_column])
print('Measured STD of median bias : {0:4.6f} e'.format(measured_std_medianbias))
median_error = np.std(stack, axis=0) /np.sqrt(n_images)
median_pixel_error = np.median(median_error)
print('Median STD of each pixel : {0:4.6f} e'.format(median_pixel_error))

"""
We can see that the standard deviation computed on the median bias is slighlty higher than the
associated error computed through the other techniques. We already analyzed the origin of this
disagreement.
Finally, we can plot the distribution of the error associated to each pixel of the median bias with
the different estimates of the error. This plot should guide you towards the best choice fro the error
to be associated to the median bias.
"""

# Standard deviation of each pixel
STD_pixel = np.std(stack, axis=0)
plt.figure(figsize=(8,6))
plt.hist(median_error.flatten(), bins=20, range=(0,2.5), density=True,histtype='step', label='Pixel-based error')
plt.axvline(expected_noise_medianbias, c='C1', label='Error using readoutnoise')
plt.axvline(expected_std_medianbias, c='C2', label='Expected error using biasSTD')
plt.axvline(measured_std_medianbias, c='C3', label='Measured STD of medianbias')
plt.axvline(median_pixel_error, c='C4', label='Average Pixel-based error')
plt.xlabel('e')
plt.ylabel('Density')
plt.legend()
plt.show()

# =============================================================================
print("\n\n1.8 Saving the output")
print("=============================================================================\n")

"""
1.8 Saving the output

We save our files using the pickle module. This model allows the storage on disk of arrays as well
as objects and dictionaries.
"""

with open(f'{taste_dir}/bias/median_bias.p', 'wb', buffering=0) as f:
    # noinspection PyTypeChecker
    pickle.dump(median_bias, f)
with open(f'{taste_dir}/bias/median_bias_error.p', 'wb', buffering=0) as f:
    # noinspection PyTypeChecker
    pickle.dump(median_error, f)
with open(f'{taste_dir}/bias/median_bias_error_value.p', 'wb', buffering=0) as f:
    # noinspection PyTypeChecker
    pickle.dump(median_pixel_error, f)
with open(f'{taste_dir}/bias/stack_bias.p', 'wb', buffering=0) as f:
    # noinspection PyTypeChecker
    pickle.dump(stack, f)

# load the data
median_bias = pickle.load(open(f'{taste_dir}/bias/median_bias.p', 'rb'))

print('Median bias frame:')
print(median_bias)
