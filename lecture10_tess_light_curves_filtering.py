import numpy as np
import pickle
import matplotlib.pyplot as plt
from astropy.io import fits

# Attempt to import wotan; provide user-friendly error if missing
try:
    from wotan import flatten, transit_mask
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Required package 'wotan' not found. Install it via `pip install wotan` and retry.")

# =============================================================================
# 10 Filtering TESS Light Curves using Wotan
# =============================================================================
# This script performs detrending of TESS SAP and PDCSAP light curves
# and generates all diagnostic plots as in the lesson notebook.

# 1. Configuration -----------------------------------------------------------
print("# 1. Loading selected TESS light curve data")

# Paths (keep original structure)
tess_dir = 'TESS_analysis'
sector_file = f"{tess_dir}/qatar1_TESS_sector024_selected.p"
print(f"Loading data from: {sector_file}")
with open(sector_file, 'rb') as f:
    data = pickle.load(f)

time = data['time']
sap_flux = data['sap_flux']
sap_flux_error = data['sap_flux_error']
pdcsap_flux = data['pdcsap_flux']
pdcsap_flux_error = data['pdcsap_flux_error']
print(f"Loaded {len(time)} points")

# Transit parameters (Qatar-1b ephemeris)
# Mid-transit time and orbital period must be updated for Qatar-1
Transit_time = 2458977.55982   # BJD_TDB approximate mid-transit from TESS Sector 24 data
Period = 1.42002504            # Orbital period [days] for Qatar-1b
# Transit duration (from ExoFOP): adjust if necessary
Transit_duration_hrs = 2.982    # hours
Transit_window = Transit_duration_hrs * 2 / 24  # days (double duration)

# Mask in-transit points
mask = transit_mask(time=time, period=Period, duration=Transit_window, T0=Transit_time)
print(f"Masked {np.sum(mask)} in-transit points")
# 2. HSpline flattening ------------------------------------------------------
sap_flat, sap_trend = flatten(time, sap_flux, method='hspline', window_length=0.5,
                              break_tolerance=0.5, return_trend=True)
pdcsap_flat, pdcsap_trend = flatten(time, pdcsap_flux, method='hspline',
                                    window_length=0.5, break_tolerance=0.5,
                                    return_trend=True)
sap_flat_m, sap_trend_m = flatten(time, sap_flux, method='hspline', window_length=0.5,
                                  break_tolerance=0.5, return_trend=True, mask=mask)
pdcsap_flat_m, pdcsap_trend_m = flatten(time, pdcsap_flux, method='hspline',
                                        window_length=0.5, break_tolerance=0.5,
                                        return_trend=True, mask=mask)
print("HSpline flattening done (w=0.5) with and without mask")

# 3. Plot raw & trend --------------------------------------------------------
plt.figure(figsize=(8,4))
plt.title('TESS SAP: raw & HSpline trend (no mask)')
plt.scatter(time, sap_flux, c='C0', s=3)
plt.errorbar(time, sap_flux, yerr=sap_flux_error, fmt='none', ecolor='gray', alpha=0.3)
plt.plot(time, sap_trend, c='C1', label='trend')
plt.xlabel('BJD_TDB'); plt.ylabel('SAP flux [e-/s]')
plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
plt.title('TESS SAP: normalized (no mask)')
plt.scatter(time, sap_flat, c='C0', s=3)
plt.errorbar(time, sap_flat, yerr=sap_flux_error/sap_trend, fmt='none', ecolor='gray', alpha=0.3)
plt.axhline(1, c='C1'); plt.xlabel('BJD_TDB'); plt.ylabel('Normalized flux')
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
plt.title('TESS PDCSAP: raw & HSpline trend (no mask)')
plt.scatter(time, pdcsap_flux, c='C2', s=3)
plt.errorbar(time, pdcsap_flux, yerr=pdcsap_flux_error, fmt='none', ecolor='gray', alpha=0.3)
plt.plot(time, pdcsap_trend, c='C3', label='trend')
plt.xlabel('BJD_TDB'); plt.ylabel('PDCSAP flux [e-/s]')
plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
plt.title('TESS PDCSAP: normalized (no mask)')
plt.scatter(time, pdcsap_flat, c='C2', s=3)
plt.errorbar(time, pdcsap_flat, yerr=pdcsap_flux_error/pdcsap_trend, fmt='none', ecolor='gray', alpha=0.3)
plt.axhline(1, c='C3'); plt.xlabel('BJD_TDB'); plt.ylabel('Normalized flux')
plt.tight_layout(); plt.show()

# 4. Phase-folded normalized ------------------------------------------------
phase = (time - Transit_time - Period/2) % Period - Period/2
# Optional shift for phase plots (in days)
phase_shift = 0.0  # adjust >0 to move points right, <0 to move left
phase_plot = phase + phase_shift

plt.figure(figsize=(8,4))
plt.title('SAP: phase-folded normalized (no mask)')
plt.scatter(phase_plot, sap_flat, c='C0', s=3)
plt.errorbar(phase_plot, sap_flat, yerr=sap_flux_error/sap_trend,
             fmt='none', ecolor='gray', alpha=0.3)
plt.axhline(1, c='C1'); plt.xlim(-0.4 + phase_shift, 0.4 + phase_shift)
plt.xlabel('Phase [d]'); plt.ylabel('Normalized flux')
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
plt.title('PDCSAP: phase-folded normalized (no mask)')
plt.scatter(phase_plot, pdcsap_flat, c='C2', s=3)
plt.errorbar(phase_plot, pdcsap_flat, yerr=pdcsap_flux_error/pdcsap_trend,
             fmt='none', ecolor='gray', alpha=0.3)
plt.axhline(1, c='C3'); plt.xlim(-0.4 + phase_shift, 0.4 + phase_shift)
plt.xlabel('Phase [d]'); plt.ylabel('Normalized flux')
plt.tight_layout(); plt.show()

# 5. Phase-folded normalized with masked points -------------------------------- Phase-folded normalized with masked points --------------------------------
plt.figure(figsize=(8,4))
plt.title('SAP: phase-folded normalized with masked points')
# plot normalized SAP flattened light curve\ nplt.scatter(phase, sap_flat, s=2, c='C0', alpha=0.5, label='normalized')
# overlay masked in-transit points on normalized curve
plt.scatter(phase[mask], sap_flat[mask], s=30, facecolors='none', edgecolors='r', label='masked')
plt.axhline(1, c='k', ls='--')
plt.xlim(-0.4,0.4)
plt.xlabel('Phase [d]'); plt.ylabel('Normalized flux')
plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
plt.title('PDCSAP: phase-folded normalized with masked points')
# plot normalized PDCSAP flattened light curve
plt.scatter(phase, pdcsap_flat, s=2, c='C2', alpha=0.5, label='normalized')
# overlay masked in-transit points
plt.scatter(phase[mask], pdcsap_flat[mask], s=30, facecolors='none', edgecolors='r', label='masked')
plt.axhline(1, c='k', ls='--')
plt.xlim(-0.4,0.4)
plt.xlabel('Phase [d]'); plt.ylabel('Normalized flux')
plt.legend(); plt.tight_layout(); plt.show()

# 6. Compare trends no mask vs mask ----------------------------------------- Compare trends no mask vs mask -----------------------------------------
plt.figure(figsize=(8,4))
plt.title('SAP trend: no mask vs mask')
plt.scatter(time, sap_flux, s=2, c='C0', alpha=0.5)
plt.plot(time, sap_trend, c='C1', label='no mask')
plt.plot(time, sap_trend_m, c='C2', label='mask')
plt.xlabel('BJD_TDB'); plt.ylabel('Flux [e-/s]')
plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
plt.title('PDCSAP trend: no mask vs mask')
plt.scatter(time, pdcsap_flux, s=2, c='C2', alpha=0.5)
plt.plot(time, pdcsap_trend, c='C3', label='no mask')
plt.plot(time, pdcsap_trend_m, c='C4', label='mask')
plt.xlabel('BJD_TDB'); plt.ylabel('Flux [e-/s]')
plt.legend(); plt.tight_layout(); plt.show()

# 7. Phase-folded normalized masked ------------------------------------------
plt.figure(figsize=(8,4))
plt.title('SAP: phase-folded normalized (mask)')
plt.scatter(phase, sap_flat_m, c='C0', s=3)
plt.errorbar(phase, sap_flat_m, yerr=sap_flux_error/sap_trend_m, fmt='none', ecolor='gray', alpha=0.3)
plt.axhline(1, c='C1'); plt.xlim(-0.4,0.4)
plt.xlabel('Phase [d]'); plt.ylabel('Normalized flux')
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
plt.title('PDCSAP: phase-folded normalized (mask)')
plt.scatter(phase, pdcsap_flat_m, c='C2', s=3)
plt.errorbar(phase, pdcsap_flat_m, yerr=pdcsap_flux_error/pdcsap_trend_m, fmt='none', ecolor='gray', alpha=0.3)
plt.axhline(1, c='C3'); plt.xlim(-0.4,0.4)
plt.xlabel('Phase [d]'); plt.ylabel('Normalized flux')
plt.tight_layout(); plt.show()

# 8. Test various algorithms and window lengths -----------------------------
methods = [('hspline',0.5), ('biweight',1.0), ('biweight',1.5), ('hspline',1.0), ('hspline',1.5)]
stats = {}
for method,w in methods:
    f_s, t_s = flatten(time, sap_flux, method=method, window_length=w,
                       break_tolerance=0.5, return_trend=True, mask=mask)
    f_p, t_p = flatten(time, pdcsap_flux, method=method, window_length=w,
                       break_tolerance=0.5, return_trend=True, mask=mask)
    stats[(method,w)] = (np.std(f_s[~mask]), np.std(f_p[~mask]))
    print(f"STD SAP {method} w={w}: {stats[(method,w)][0]:.6f}")
    print(f"STD PDCSAP {method} w={w}: {stats[(method,w)][1]:.6f}")

# 9. Multi-trend comparison -------------------------------------------------
plt.figure(figsize=(8,4))
plt.title('SAP: multiple trend models (no mask, mask, biweight)')
plt.scatter(time, sap_flux, s=2, c='C0', alpha=0.5)
plt.plot(time, sap_trend, c='C1', label='hspline w=0.5 no mask')
plt.plot(time, sap_trend_m, c='C2', label='hspline w=0.5 mask')
# pick biweight w=1.0
_, sap_trend_bw10 = flatten(time, sap_flux, method='biweight', window_length=1.0,
                            break_tolerance=0.5, return_trend=True, mask=mask)
plt.plot(time, sap_trend_bw10, c='C3', label='biweight w=1.0 mask')
plt.xlabel('BJD_TDB'); plt.ylabel('Flux [e-/s]')
plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
plt.title('PDCSAP: multiple trend models (no mask, mask, biweight)')
plt.scatter(time, pdcsap_flux, s=2, c='C2', alpha=0.5)
plt.plot(time, pdcsap_trend, c='C3', label='hspline w=0.5 no mask')
plt.plot(time, pdcsap_trend_m, c='C4', label='hspline w=0.5 mask')
_, pdcsap_trend_bw10 = flatten(time, pdcsap_flux, method='biweight', window_length=1.0,
                              break_tolerance=0.5, return_trend=True, mask=mask)
plt.plot(time, pdcsap_trend_bw10, c='C5', label='biweight w=1.0 mask')
plt.xlabel('BJD_TDB'); plt.ylabel('Flux [e-/s]')
plt.legend(); plt.tight_layout(); plt.show()

# 10. Save filtered data ----------------------------------------------------
output = {
    'time': time,
    'sap_flat': sap_flat_m, 'sap_err': sap_flux_error/sap_trend_m,
    'pdcsap_flat': pdcsap_flat_m, 'pdcsap_err': pdcsap_flux_error/pdcsap_trend_m
}
outfile = f"{tess_dir}/qatar1_TESS_sector024_filtered.p"
with open(outfile, 'wb') as f:
    pickle.dump(output, f)
print(f"Saved filtered data to: {outfile}")
print("Done.")