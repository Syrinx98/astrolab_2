#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
8 Accessing and Understanding TESS Data Products

This script demonstrates how to:
1. Understand what TESS Target Pixel Files (TPFs) and Light Curve (LC) files are, and how they differ.
2. Explore the contents of a TPF:
   - Primary HDU (file-wide metadata)
   - PIXELS extension (time-series of raw & calibrated pixel images)
   - APERTURE extension (bitmask describing pixel usage)
   - TARGET COSMIC RAY extension (cosmic‐ray corrections)
3. Use the WCS (World Coordinate System) to plot pixel stamps in RA/Dec coordinates.
4. Decode and interpret aperture bitmask values.
5. Compare TPFs to LC files and plot the light curves.
6. Ensure clear, informative printouts and well-formatted, readable plots.

Keep all file paths exactly as in the original code to match the author's directory structure.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS

# =============================================================================
# 1. File paths and overview
# =============================================================================
tess_dir        = "TESS_analysis"
sector24_tpf    = f"{tess_dir}/qatar_1_sector24_tp.fits"
sector25_tpf    = f"{tess_dir}/qatar_1_sector25_tp.fits"
sector24_lc     = f"{tess_dir}/qatar_1_sector24_lc.fits"
sector25_lc     = f"{tess_dir}/qatar_1_sector25_lc.fits"

print("\n=== 1. Files being used ===")
print(f"  • Sector 24 TPF: {sector24_tpf}")
print(f"  • Sector 25 TPF: {sector25_tpf}")
print(f"  • Sector 24 LC : {sector24_lc}")
print(f"  • Sector 25 LC : {sector25_lc}")

# =============================================================================
# 2. Inspect TPF structure without loading full data
# =============================================================================
print("\n=== 2. TPF file structure (fits.info) ===")
fits.info(sector24_tpf)

with fits.open(sector24_tpf) as tphdu24:
    print("\n-- HDU List for Sector 24 TPF --")
    tphdu24.info()
    print("\nColumns in the PIXELS extension (HDU=1):")
    print(tphdu24[1].columns)

# =============================================================================
# 3. Read and plot the first valid FLUX image for Sector 24 with WCS overlay
# =============================================================================
print("\n=== 3. Plotting first valid FLUX image (Sector 24) ===")
with fits.open(sector24_tpf) as tphdu:
    pixel_table = tphdu[1].data
    # Find the first cadence where FLUX is finite everywhere
    valid_idx = next(
        i for i in range(len(pixel_table['FLUX']))
        if np.isfinite(pixel_table['FLUX'][i]).all()
    )
    first_image = pixel_table['FLUX'][valid_idx]
    print(f"  • Using cadence index {valid_idx} with shape {first_image.shape}")
    wcs = WCS(tphdu[2].header)

fig = plt.figure(figsize=(7, 7))
ax  = fig.add_subplot(111, projection=wcs)
im  = ax.imshow(first_image, origin='lower', cmap=plt.get_cmap('cividis'))
ax.set_xlabel('Right Ascension (RA)')
ax.set_ylabel('Declination (Dec)')
ax.set_title("Sector 24 TPF: First Valid Flux Image")
ax.grid(color='white', ls='solid')
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Flux (e⁻/s)")
fig.tight_layout()
plt.show()
plt.close(fig)

# =============================================================================
# 4. Repeat for Sector 25 with higher DPI and explicit colorbar
# =============================================================================
print("\n=== 4. Plotting first valid FLUX image (Sector 25) ===")
with fits.open(sector25_tpf) as tphdu:
    pixel_table = tphdu[1].data
    valid_idx = next(
        i for i in range(len(pixel_table['TIME']))
        if np.isfinite(pixel_table['FLUX'][i]).all()
    )
    first_image = pixel_table['FLUX'][valid_idx]
    print(f"  • Sector 25: using cadence index {valid_idx}")
    wcs = WCS(tphdu[2].header)

fig = plt.figure(figsize=(7, 7), dpi=300)
ax  = fig.add_subplot(111, projection=wcs)
im  = ax.imshow(first_image, origin='lower', cmap=plt.get_cmap('cividis'))
ax.set_xlabel('RA')
ax.set_ylabel('Dec')
ax.set_title("Sector 25 TPF: First Valid Flux Image")
ax.grid(color='white', ls='solid')
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Flux (e⁻/s)")
fig.tight_layout()
plt.show()
plt.close(fig)

# =============================================================================
# 5. Inspect the Aperture Mask (bitmask) for Sector 24
# =============================================================================
print("\n=== 5. Aperture Mask (HDU=2) ===")
with fits.open(sector24_tpf) as tphdu:
    aperture_mask = tphdu[2].data
    print(f"  • Aperture mask shape: {aperture_mask.shape}, dtype: {aperture_mask.dtype}")
    print("  • Sample bitmask values:\n", aperture_mask)

fig, ax = plt.subplots(figsize=(6, 6))
cb = ax.imshow(aperture_mask, origin='lower', cmap='hot')
ax.set_title("Sector 24 Aperture Bitmask")
cbar = fig.colorbar(cb)
cbar.set_label("Bitmask Value")
fig.tight_layout()
plt.show()
plt.close(fig)

# =============================================================================
# 6. Decode and interpret bitmask values
# =============================================================================
print("\n=== 6. Bitmask decoding examples ===")
# Definitions (from TESS Aperture Mask Image Flags)
bit_flags = {
    0: "Collected by spacecraft",
    1: "Used in optimal photometric aperture (SAP_FLUX)",
    2: "Used for flux-weighted centroid",
    3: "Used for PSF-based centroid",
    6: "On CCD output B",
    # ... add other flags as needed
}

def decode_bitmask(val: int):
    """Return which bits are set and their meanings."""
    bits = list(np.binary_repr(val, width=8))[::-1]  # LSB first
    set_indices = [i for i, b in enumerate(bits) if b == '1']
    descriptions = [bit_flags.get(i, f"Unknown flag {i}") for i in set_indices]
    return set_indices, descriptions

for test_val in (75, 69):
    idxs, descs = decode_bitmask(test_val)
    print(f"  • Value {test_val}: bits set {idxs} -> {descs}")

# Check optimal aperture bit (bit 1 -> mask value 2)
val = 75
mask_value = 1 << 1
if np.bitwise_and(val, mask_value):
    print(f"  • Pixel value {val} has the optimal-aperture bit set.")
else:
    print(f"  • Pixel value {val} does NOT have the optimal-aperture bit set.")

# =============================================================================
# 7. Inspect the TARGET COSMIC RAY extension (HDU=3)
# =============================================================================
print("\n=== 7. TARGET COSMIC RAY extension ===")
with fits.open(sector24_tpf) as tphdu:
    cr_hdu = tphdu[3]
    print(f"  • HDU name: {cr_hdu.name}")
    print(f"  • Columns: {cr_hdu.columns.names}")
    cr_data = cr_hdu.data
    print(f"  • Number of cosmic-ray events: {len(cr_data)}")
    if len(cr_data) > 0:
        print("  • Sample records:")
        for row in cr_data[:5]:
            print("    ", dict(zip(cr_hdu.columns.names, row)))

# =============================================================================
# 8. Compare to Light Curve (LC) files
# =============================================================================
def inspect_and_plot_lc(lc_path: str, sector: int):
    """
    Open the LC file, print its structure, and plot SAP_FLUX vs TIME.
    """
    print(f"\n--- Light Curve file (Sector {sector}) ---")
    fits.info(lc_path)
    with fits.open(lc_path) as lchdu:
        cols = lchdu[1].columns.names
        print("  • Available columns:", cols)

        data = lchdu[1].data
        time = data['TIME']

        if 'SAP_FLUX' in cols:
            sap_flux = data['SAP_FLUX']
        elif 'PDCSAP_FLUX' in cols:
            sap_flux = data['PDCSAP_FLUX']
        else:
            raise KeyError("No SAP_FLUX or PDCSAP_FLUX column found in LC file.")

        # Mask invalid entries
        valid = np.isfinite(time) & np.isfinite(sap_flux)
        print(f"  • Plotting {valid.sum()} valid points out of {len(time)} total.")

    # Generate light curve plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time[valid], sap_flux[valid], '.', markersize=2, linestyle='-')
    ax.set_xlabel("BJD - 2,457,000 (days)")
    ax.set_ylabel("SAP Flux (e⁻/s)")
    ax.set_title(f"TESS Sector {sector} Light Curve")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()
    plt.close(fig)

inspect_and_plot_lc(sector24_lc, sector=24)
inspect_and_plot_lc(sector25_lc, sector=25)

print("\nAll steps completed successfully.")
