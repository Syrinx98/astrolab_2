"""
8 Accessing and Understanding TESS Data Products

In this code, we demonstrate how to:

1. Understand what TESS Target Pixel Files (TPFs) and Light Curve files are, and how they differ.
2. Explore the contents of a TPF:
   - Primary HDU
   - PIXELS extension with raw and calibrated pixels
   - APERTURE extension with information about pixel usage (bitmasks)
   - TARGET COSMIC RAY extension
3. Use the WCS (World Coordinate System) to plot the TPF data in RA/Dec coordinates.
4. Understand how to interpret pixel values in the aperture extension, which are stored as bitmasks.

We assume that you have already downloaded and saved the TPF and LC files with shorter names,
following the instructions in the Moodle text. We also assume you are inside a conda environment that
can handle the required packages (astropy, matplotlib, numpy).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from astropy.io import fits
from astropy.wcs import WCS

# =============================================================================
print("\n\n 5.1. Reading TESS Target Pixel Files")
print("=============================================================================\n")

"""
We define the names of the TPF files. 
Replace these names with the actual names of your TESS TPF files if different.
"""

tess_dir = "TESS_analysis"

sector24_tpf = f"{tess_dir}/qatar_1_sector24_tp.fits"
sector25_tpf = f"{tess_dir}/qatar_1_sector25_tp.fits"

# Just print some info about the files
print("TPF files being used:")
print("Sector 44:", sector24_tpf)


# =============================================================================
print("\n\n 5.2. Inspecting the TPF structure")
print("=============================================================================\n")

"""
We can use fits.info to see the structure of the FITS file without loading everything.
"""

fits.info(sector24_tpf)

# Open the TPF file
tphdu = fits.open(sector24_tpf)

print("\n\nHDU List for sector24_tpf:")
print(tphdu.info())

# The first extension (HDU=1) contains the PIXELS data
print("\nColumns in the PIXELS extension:")
print(tphdu[1].columns)


# =============================================================================
print("\n\n 5.3. Reading and plotting the first image in the FLUX column with WCS overlay")
print("=============================================================================\n")

"""
We read the first FLUX image and plot it with a WCS projection. The WCS is stored in the aperture HDU (HDU=2).
"""

tpf_data = tphdu[1].data
first_image = tpf_data['FLUX'][0]

# The WCS is the same in the aperture extension (HDU=2)
wcs = WCS(tphdu[2].header)

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection=wcs)
im = ax.imshow(first_image, origin='lower', cmap=plt.colormaps['cividis'])
ax.set_xlabel('RA', fontsize=14)
ax.set_ylabel('Dec', fontsize=14)
ax.grid(axis='both', color='white', ls='solid')

plt.title("First FLUX image from Sector 44 TPF")
plt.show()


# =============================================================================
print("\n\n 5.4. Checking another sector and adding a colorbar")
print("=============================================================================\n")

"""
Sometimes the first image might be invalid (NaN or Inf). We loop over TIME steps until we find a valid image.
"""

tphdu = fits.open(sector25_tpf)
tpf_data = tphdu[1].data

for i_check in range(len(tpf_data['TIME'])):
    if np.isfinite(tpf_data['FLUX'][i_check][0,0]):
        break

first_image = tpf_data['FLUX'][i_check]
wcs = WCS(tphdu[2].header)

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection=wcs)
im = ax.imshow(first_image, origin='lower', cmap=plt.colormaps['cividis'])
ax.set_xlabel('RA', fontsize=12)
ax.set_ylabel('Dec', fontsize=12)
ax.grid(axis='both', color='white', ls='solid')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.title(f"Valid FLUX image from Sector 46 TPF (index={i_check})")
plt.show()


# =============================================================================
print("\n\n 5.5. Inspecting the Aperture Mask")
print("=============================================================================\n")

"""
The aperture extension (HDU=2) contains integer bitmasks describing the status of each pixel.
We load it and print it.
"""

tphdu = fits.open(sector24_tpf)
aperture = tphdu[2].data
print("Aperture Mask Array:")
print(aperture)

# We plot the aperture bitmask
fig, ax = plt.subplots(figsize=(6,6))
cbx = ax.imshow(aperture, cmap=plt.cm.hot, origin="lower", alpha=1.0)
fig.suptitle("Aperture bitmask for Sector 44", fontsize=14)
cbar = fig.colorbar(cbx)
plt.show()

# =============================================================================
print("\n\n 5.6. Understanding Bitmasks")
print("=============================================================================\n")

"""
Each pixel is assigned a bitmask. For example, a pixel value of 75 in decimal:
75 decimal = binary 1001011

We can decode the bits with np.binary_repr.

If we want to select pixels used in the optimal photometric aperture (bit value 2), we must do a bitwise AND 
between the aperture mask and the value 2. If result >0 means that bit was set.
"""

test_value = 75
bits = np.binary_repr(test_value)
print(f"Value {test_value} in binary: {bits}")

# Example of bitwise AND
test_value2 = 69
bits2 = np.binary_repr(test_value2)
print(f"Value {test_value2} in binary: {bits2}")

# Let's say we want to check if a pixel is in the optimal aperture (bit = 2)
# The optimal aperture bit is 2 decimal (binary 10)
optimal_bit = 2
bitwise_result = np.bitwise_and(test_value, optimal_bit)
print(f"Bitwise AND of {test_value} and {optimal_bit} = {bitwise_result}")
if bitwise_result > 0:
    print("The pixel was used in the optimal aperture.")
else:
    print("The pixel was NOT used in the optimal aperture.")


# Done
print("\nAll steps completed.")
