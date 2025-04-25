#!/usr/bin/env python3
"""
improved_tess_lightcurve_analysis.py

This script demonstrates the extraction and cleaning of TESS light curves (LCf) for a single sector,
following the lesson on TESS SAP and PDCSAP photometry and quality flag handling.

Steps:
1. Load and inspect the FITS file structure.
2. Extract photometry arrays and convert times.
3. Plot raw flux vs BJD_TDB for SAP and PDCSAP separately.
4. Initial SAP vs PDCSAP comparison plot.
5. Print array shapes and sample PDCSAP values.
6. Conservative quality filtering (NaN/Inf and QUALITY>0) and combined selection overview.
7. Optional selective filtering via custom bitmask.
8. Segment markers for visual checks.
9. Manual exclusion of specified time ranges with detailed plot.
10. Save cleaned data for later use.

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pickle

# =============================================================================
# 1. Configuration and File Paths
# =============================================================================
tess_dir = 'TESS_analysis'
sector = 24
lcf_filename = f'qatar_1_sector{sector}_lc.fits'
sector24_lcf = os.path.join(tess_dir, lcf_filename)

print(f"\n{'='*80}")
print(f"1. Loading FITS: {sector24_lcf}")
print(f"{'='*80}\n")

if not os.path.isfile(sector24_lcf):
    raise FileNotFoundError(f"File not found: {sector24_lcf}")

# =============================================================================
# 2. Inspect and extract data
# =============================================================================
print("Inspecting FITS structure:")
fits.info(sector24_lcf)

with fits.open(sector24_lcf, mode='readonly') as hdulist:
    lc_hdu = hdulist[1]
    data   = lc_hdu.data.copy()
    header = lc_hdu.header.copy()

# Photometry and quality arrays
time_btjd   = data['TIME']
sap_flux    = data['SAP_FLUX']
sap_err     = data['SAP_FLUX_ERR']
pdcsap_flux = data['PDCSAP_FLUX']
pdcsap_err  = data['PDCSAP_FLUX_ERR']
quality     = data['QUALITY']

# Convert BTJD to BJD_TDB
bjdrefi = header.get('BJDREFI', 2457000)
bjdreff = header.get('BJDREFF', 0.0)
time_bjd = time_btjd + bjdrefi + bjdreff

print(f"Data counts: TIME={time_bjd.size}, SAP={sap_flux.size}, PDCSAP={pdcsap_flux.size}, Q={quality.size}")

# =============================================================================
# 3. Plot raw SAP and PDCSAP separately vs BJD_TDB
# =============================================================================
plt.figure(figsize=(8,4), dpi=120)
plt.plot(time_bjd, sap_flux,    '.', markersize=4, label='SAP Raw')
plt.xlabel('BJD_TDB [days]')
plt.ylabel('SAP Flux [e-/s]')
plt.title(f'Sector {sector}: SAP Flux vs BJD_TDB')
plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4), dpi=120)
plt.plot(time_bjd, pdcsap_flux, '.', markersize=4, label='PDCSAP Raw')
plt.xlabel('BJD_TDB [days]')
plt.ylabel('PDCSAP Flux [e-/s]')
plt.title(f'Sector {sector}: PDCSAP Flux vs BJD_TDB')
plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# =============================================================================
# 4. Initial SAP vs PDCSAP plot
# =============================================================================
plt.figure(figsize=(8,4), dpi=120)
plt.scatter(time_bjd, sap_flux,    s=6, label='SAP')
plt.scatter(time_bjd, pdcsap_flux, s=6, label='PDCSAP')
plt.xlabel('BJD_TDB [days]')
plt.ylabel('Flux [e-/s]')
plt.title(f'Sector {sector}: SAP vs PDCSAP')
plt.legend(); plt.grid(alpha=0.4)
plt.tight_layout(); plt.show()
print('Non-finite PDCSAP indices:', np.where(~np.isfinite(pdcsap_flux))[0][:10])

# =============================================================================
# 5. Print shapes and sample PDCSAP values
# =============================================================================
print(f"Number of BJD epochs  : {time_bjd.shape}")
print(f"Number of SAP epochs  : {sap_flux.shape}")
print(f"Number of PDCSAP epochs: {pdcsap_flux.shape}")
print('Some elements of PDCSAP flux [10:20]:', pdcsap_flux[10:20])

# =============================================================================
# 6. Conservative filtering and overview plot
# =============================================================================
finite = np.isfinite(pdcsap_flux)
good_q = (quality == 0)
mask_cons = finite & good_q
print(f"Conservative: {mask_cons.sum()} / {len(mask_cons)} points kept.")

plt.figure(figsize=(8,4), dpi=120)
plt.scatter(time_bjd[mask_cons], sap_flux[mask_cons],    s=6, label='SAP - selected')
plt.scatter(time_bjd,               pdcsap_flux,          s=6, label='PDCSAP')
plt.scatter(time_bjd[~mask_cons],   sap_flux[~mask_cons],  s=6, c='r', alpha=0.6, label='SAP - excluded')
plt.errorbar(time_bjd[mask_cons], sap_flux[mask_cons], yerr=sap_err[mask_cons], fmt='none', alpha=0.5, ecolor='k', zorder=-1)
plt.xlabel('BJD_TDB'); plt.ylabel('e-/s'); plt.title(f'Sector {sector}: conservative selection overview')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# =============================================================================
# 7. Selective filtering (optional)
# =============================================================================
flags = [1,2,3,4,5,6,8,10,13,15]
ref_mask = sum(2**(f-1) for f in flags)
mask_sel = finite & (~(np.bitwise_and(quality, ref_mask)>0))
print(f"Selective: kept {mask_sel.sum()} points (ref_mask={ref_mask}).")

# =============================================================================
# 8. Segment markers
# =============================================================================
min_t, max_t = time_bjd[mask_cons].min(), time_bjd[mask_cons].max()
segs = np.linspace(min_t, max_t, 31)
fig, ax = plt.subplots(figsize=(8,3), dpi=120)
ax.scatter(time_bjd[mask_cons], sap_flux[mask_cons], s=6)
for i, x in enumerate(segs,1):
    ax.axvline(x, linestyle='--', alpha=0.3)
    print(f"Segment {i}: {x:.5f}")
ax.set(xlabel='BJD_TDB', ylabel='Flux'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# =============================================================================
# 9. Manual exclusion — definizione di cutoff e mask_final
# =============================================================================
# punto di taglio (esempio preso dal tuo secondo script)
cutoff = 2458961.0912

# mask_final: teniamo solo i punti che superano il cutoff, a partire da quelli già in mask_cons
mask_final = mask_cons & (time_bjd > cutoff)

print(f"Manual: {mask_final.sum()} punti finali su {len(mask_cons)} iniziali conservativi.")

# =============================================================================
# 9b. Manual exclusion — plot finale (corretto)
# =============================================================================
plt.figure(figsize=(8,4), dpi=300)

# SAP selezionati (conservative selection)
plt.scatter(time_bjd[mask_cons], sap_flux[mask_cons],
            s=6, label='SAP - selected data')

# PDCSAP grezzi
plt.scatter(time_bjd, pdcsap_flux,
            s=6, label='PDCSAP')

# SAP esclusi per qualità
plt.scatter(time_bjd[~mask_cons], sap_flux[~mask_cons],
            s=6, c='r', label='SAP - excluded data')

# SAP esclusi manualmente (tra quelli inizialmente validi)
plt.scatter(time_bjd[~mask_final & mask_cons],
            sap_flux[~mask_final & mask_cons],
            s=20, c='y', marker='x', label='SAP - manually excluded')

# Barre di errore sui punti conservatively selezionati
plt.errorbar(time_bjd[mask_cons], sap_flux[mask_cons],
             yerr=sap_err[mask_cons], fmt=' ',
             alpha=0.5, ecolor='k', zorder=-1)

plt.xlabel('BJD_TDB [d]')
plt.ylabel('Flux [e⁻/s]')
plt.title(f'TESS Lightcurve for qatar1 – sector {sector} (final selection)', fontsize=12)

# Limiti di zoom: da inizio selezione conservativa fino al cutoff
plt.xlim(min_t, cutoff)
# Range di flusso scelta “a mano” per evidenziare la zona di interesse
plt.ylim(2300, 2450)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


