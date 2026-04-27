import numpy as np

acq = "2025-05-30_050"

base = "outputs/04_radiometric_calibration_qc"

valid_idx = np.load(f"{base}/{acq}/valid_band_indices.npy")
valid_wl = np.load(f"{base}/{acq}/valid_wavelengths_nm.npy")

print("Bande valide:", len(valid_idx))
print("Range valido:", valid_wl.min(), "-", valid_wl.max())

print("\nPrime bande valide:")
print(valid_wl[:10])

print("\nUltime bande valide:")
print(valid_wl[-10:])

# Bande escluse
total_bands = 204
excluded_idx = np.setdiff1d(np.arange(total_bands), valid_idx)

print("\nNumero bande escluse:", len(excluded_idx))
print("Prime escluse (index):", excluded_idx[:10])
print("Ultime escluse (index):", excluded_idx[-10:])