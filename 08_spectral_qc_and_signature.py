from pathlib import Path
import csv
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


# ============================================================
# CONFIG
# ============================================================

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# SAVE
# ============================================================

def save_csv(records, output_path):
    if not records:
        return

    keys = list(records[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def save_json(payload, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ============================================================
# SAFE HELPERS
# ============================================================

def safe_float(value):
    if value is None:
        return None

    try:
        if np.isnan(value) or np.isinf(value):
            return None
    except TypeError:
        return None

    return float(value)


def safe_nanpercentile(spectra, percentile, axis=0):
    """
    Percentile robusto con NaN.
    Se una banda è tutta NaN, restituisce NaN per quella banda.
    """
    with np.errstate(all="ignore"):
        return np.nanpercentile(spectra, percentile, axis=axis)


def safe_nanmean(spectra, axis=0):
    with np.errstate(all="ignore"):
        return np.nanmean(spectra, axis=axis)


def safe_nanmedian(spectra, axis=0):
    with np.errstate(all="ignore"):
        return np.nanmedian(spectra, axis=axis)


def safe_nanstd(spectra, axis=0):
    with np.errstate(all="ignore"):
        return np.nanstd(spectra, axis=axis)


# ============================================================
# STATISTICS
# ============================================================

def compute_spectral_statistics(spectra, wavelengths):
    records = []

    mean = safe_nanmean(spectra, axis=0)
    median = safe_nanmedian(spectra, axis=0)
    std = safe_nanstd(spectra, axis=0)

    p01 = safe_nanpercentile(spectra, 1, axis=0)
    p05 = safe_nanpercentile(spectra, 5, axis=0)
    p25 = safe_nanpercentile(spectra, 25, axis=0)
    p50 = safe_nanpercentile(spectra, 50, axis=0)
    p75 = safe_nanpercentile(spectra, 75, axis=0)
    p95 = safe_nanpercentile(spectra, 95, axis=0)
    p99 = safe_nanpercentile(spectra, 99, axis=0)

    n_total_pixels = spectra.shape[0]
    n_valid_values_per_band = np.sum(np.isfinite(spectra), axis=0)
    n_nan_values_per_band = np.sum(np.isnan(spectra), axis=0)
    n_inf_values_per_band = np.sum(np.isinf(spectra), axis=0)

    for i, wl in enumerate(wavelengths):
        records.append({
            "band_position": int(i),
            "wavelength_nm": float(wl),

            "n_total_pixels": int(n_total_pixels),
            "n_valid_values": int(n_valid_values_per_band[i]),
            "n_nan_values": int(n_nan_values_per_band[i]),
            "n_inf_values": int(n_inf_values_per_band[i]),

            "pct_valid_values": float(100 * n_valid_values_per_band[i] / n_total_pixels),
            "pct_nan_values": float(100 * n_nan_values_per_band[i] / n_total_pixels),
            "pct_inf_values": float(100 * n_inf_values_per_band[i] / n_total_pixels),

            "mean_reflectance": safe_float(mean[i]),
            "median_reflectance": safe_float(median[i]),
            "std_reflectance": safe_float(std[i]),

            "p01_reflectance": safe_float(p01[i]),
            "p05_reflectance": safe_float(p05[i]),
            "p25_reflectance": safe_float(p25[i]),
            "p50_reflectance": safe_float(p50[i]),
            "p75_reflectance": safe_float(p75[i]),
            "p95_reflectance": safe_float(p95[i]),
            "p99_reflectance": safe_float(p99[i]),
        })

    arrays = {
        "mean": mean,
        "median": median,
        "std": std,
        "p01": p01,
        "p05": p05,
        "p25": p25,
        "p50": p50,
        "p75": p75,
        "p95": p95,
        "p99": p99,
        "n_valid_values": n_valid_values_per_band,
        "n_nan_values": n_nan_values_per_band,
        "n_inf_values": n_inf_values_per_band,
    }

    return records, arrays


def compute_dataset_summary(spectra, wavelengths, dataset_label):
    n_total_values = int(spectra.size)
    n_finite_values = int(np.isfinite(spectra).sum())
    n_nan_values = int(np.isnan(spectra).sum())
    n_inf_values = int(np.isinf(spectra).sum())

    return {
        "dataset_label": dataset_label,

        "n_pixels": int(spectra.shape[0]),
        "n_bands": int(spectra.shape[1]),

        "n_total_values": n_total_values,
        "n_finite_values": n_finite_values,
        "n_nan_values": n_nan_values,
        "n_inf_values": n_inf_values,

        "pct_finite_values": float(100 * n_finite_values / n_total_values),
        "pct_nan_values": float(100 * n_nan_values / n_total_values),
        "pct_inf_values": float(100 * n_inf_values / n_total_values),

        "first_wavelength_nm": float(wavelengths[0]),
        "last_wavelength_nm": float(wavelengths[-1]),

        "mean_reflectance_global": safe_float(safe_nanmean(spectra, axis=None)),
        "median_reflectance_global": safe_float(safe_nanmedian(spectra, axis=None)),
        "std_reflectance_global": safe_float(safe_nanstd(spectra, axis=None)),

        "p01_reflectance_global": safe_float(safe_nanpercentile(spectra, 1, axis=None)),
        "p05_reflectance_global": safe_float(safe_nanpercentile(spectra, 5, axis=None)),
        "p25_reflectance_global": safe_float(safe_nanpercentile(spectra, 25, axis=None)),
        "p50_reflectance_global": safe_float(safe_nanpercentile(spectra, 50, axis=None)),
        "p75_reflectance_global": safe_float(safe_nanpercentile(spectra, 75, axis=None)),
        "p95_reflectance_global": safe_float(safe_nanpercentile(spectra, 95, axis=None)),
        "p99_reflectance_global": safe_float(safe_nanpercentile(spectra, 99, axis=None)),
    }


# ============================================================
# PLOTS
# ============================================================

def plot_mean_p05_p95(wavelengths, arrays, output_path, title):
    plt.figure(figsize=(10, 5))
    plt.plot(wavelengths, arrays["mean"], label="Mean reflectance")

    plt.fill_between(
        wavelengths,
        arrays["p05"],
        arrays["p95"],
        alpha=0.25,
        label="P05–P95"
    )

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_mean_std(wavelengths, arrays, output_path, title):
    lower = arrays["mean"] - arrays["std"]
    upper = arrays["mean"] + arrays["std"]

    plt.figure(figsize=(10, 5))
    plt.plot(wavelengths, arrays["mean"], label="Mean reflectance")

    plt.fill_between(
        wavelengths,
        lower,
        upper,
        alpha=0.25,
        label="Mean ± SD"
    )

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_median_iqr(wavelengths, arrays, output_path, title):
    plt.figure(figsize=(10, 5))
    plt.plot(wavelengths, arrays["median"], label="Median reflectance")

    plt.fill_between(
        wavelengths,
        arrays["p25"],
        arrays["p75"],
        alpha=0.25,
        label="P25–P75"
    )

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ============================================================
# PROCESS ONE DATASET VERSION
# ============================================================

def process_dataset_version(input_npz, out_dir, acquisition_id, dataset_label):
    if not input_npz.exists():
        return None

    data = np.load(input_npz)
    spectra = data["spectra"].astype(np.float32)
    wavelengths = data["wavelengths_nm"].astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)

    records, arrays = compute_spectral_statistics(spectra, wavelengths)

    save_csv(
        records,
        out_dir / "extracted_spectra_statistics.csv"
    )

    plot_mean_p05_p95(
        wavelengths=wavelengths,
        arrays=arrays,
        output_path=out_dir / "spectral_signature_mean_p05_p95.png",
        title=f"{acquisition_id} - {dataset_label} - Mean and P05–P95"
    )

    plot_mean_std(
        wavelengths=wavelengths,
        arrays=arrays,
        output_path=out_dir / "spectral_signature_mean_std.png",
        title=f"{acquisition_id} - {dataset_label} - Mean ± SD"
    )

    plot_median_iqr(
        wavelengths=wavelengths,
        arrays=arrays,
        output_path=out_dir / "spectral_signature_median_iqr.png",
        title=f"{acquisition_id} - {dataset_label} - Median and IQR"
    )

    summary = compute_dataset_summary(
        spectra=spectra,
        wavelengths=wavelengths,
        dataset_label=dataset_label
    )

    summary["created_at"] = datetime.now().isoformat()
    summary["acquisition_id"] = acquisition_id
    summary["input_npz"] = str(input_npz)
    summary["outputs"] = {
        "statistics_csv": str(out_dir / "extracted_spectra_statistics.csv"),
        "mean_p05_p95_png": str(out_dir / "spectral_signature_mean_p05_p95.png"),
        "mean_std_png": str(out_dir / "spectral_signature_mean_std.png"),
        "median_iqr_png": str(out_dir / "spectral_signature_median_iqr.png"),
    }

    save_json(
        summary,
        out_dir / "spectral_qc_summary.json"
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    config = load_config()

    dataset_root = Path(config["dataset_root"])
    output_root = Path(config["output_dir"])

    step07_root = output_root / "07_extracted_plant_pixels"
    step08_root = output_root / "08_spectral_qc"
    step08_root.mkdir(parents=True, exist_ok=True)

    acquisitions = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    global_summary = []

    for acquisition_dir in acquisitions:
        acquisition_id = acquisition_dir.name
        print(f"\nStep 08 - spectral QC e firma spettrale: {acquisition_id}")

        acquisition_step07_dir = step07_root / acquisition_id
        acquisition_step08_dir = step08_root / acquisition_id
        acquisition_step08_dir.mkdir(parents=True, exist_ok=True)

        datasets_to_process = {
            "RAW_FULL": {
                "input_npz": acquisition_step07_dir / "plant_pixel_spectra_RAW_FULL.npz",
                "output_dir": acquisition_step08_dir / "RAW_FULL",
            },
            "CLEAN_NAN": {
                "input_npz": acquisition_step07_dir / "plant_pixel_spectra_CLEAN_NAN.npz",
                "output_dir": acquisition_step08_dir / "CLEAN_NAN",
            },
        }

        acquisition_summary = {
            "created_at": datetime.now().isoformat(),
            "acquisition_id": acquisition_id,
            "datasets": {},
            "missing_datasets": [],
        }

        for dataset_label, info in datasets_to_process.items():
            input_npz = info["input_npz"]
            out_dir = info["output_dir"]

            if not input_npz.exists():
                print(f"  ATTENZIONE: {dataset_label} non trovato: {input_npz}")
                acquisition_summary["missing_datasets"].append(dataset_label)
                continue

            print(f"  Elaboro {dataset_label}")

            summary = process_dataset_version(
                input_npz=input_npz,
                out_dir=out_dir,
                acquisition_id=acquisition_id,
                dataset_label=dataset_label,
            )

            acquisition_summary["datasets"][dataset_label] = summary

            print(f"    Pixel: {summary['n_pixels']}")
            print(f"    Bande: {summary['n_bands']}")
            print(
                f"    Range: "
                f"{summary['first_wavelength_nm']:.2f} - "
                f"{summary['last_wavelength_nm']:.2f} nm"
            )
            print(f"    NaN: {summary['pct_nan_values']:.4f}%")
            print(f"    Output: {out_dir}")

        save_json(
            acquisition_summary,
            acquisition_step08_dir / "spectral_qc_summary.json"
        )

        global_summary.append(acquisition_summary)

    save_json(
        {
            "created_at": datetime.now().isoformat(),
            "n_acquisitions": len(global_summary),
            "summaries": global_summary,
        },
        step08_root / "global_spectral_qc_summary.json"
    )

    print("\nStep 08 completato.")
    print(f"Output salvati in: {step08_root}")


if __name__ == "__main__":
    main()