from pathlib import Path
import csv
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_csv(records, output_path):
    keys = list(records[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def save_json(payload, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def compute_spectral_statistics(spectra, wavelengths):
    records = []

    mean = np.nanmean(spectra, axis=0)
    median = np.nanmedian(spectra, axis=0)
    std = np.nanstd(spectra, axis=0)

    p01 = np.nanpercentile(spectra, 1, axis=0)
    p05 = np.nanpercentile(spectra, 5, axis=0)
    p25 = np.nanpercentile(spectra, 25, axis=0)
    p75 = np.nanpercentile(spectra, 75, axis=0)
    p95 = np.nanpercentile(spectra, 95, axis=0)
    p99 = np.nanpercentile(spectra, 99, axis=0)

    for i, wl in enumerate(wavelengths):
        records.append({
            "band_position": i,
            "wavelength_nm": float(wl),
            "mean_reflectance": float(mean[i]),
            "median_reflectance": float(median[i]),
            "std_reflectance": float(std[i]),
            "p01_reflectance": float(p01[i]),
            "p05_reflectance": float(p05[i]),
            "p25_reflectance": float(p25[i]),
            "p75_reflectance": float(p75[i]),
            "p95_reflectance": float(p95[i]),
            "p99_reflectance": float(p99[i]),
        })

    arrays = {
        "mean": mean,
        "median": median,
        "std": std,
        "p01": p01,
        "p05": p05,
        "p25": p25,
        "p75": p75,
        "p95": p95,
        "p99": p99,
    }

    return records, arrays


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

        input_npz = step07_root / acquisition_id / "plant_pixel_spectra.npz"
        if not input_npz.exists():
            raise FileNotFoundError(f"File mancante: {input_npz}")

        data = np.load(input_npz)
        spectra = data["spectra"]
        wavelengths = data["wavelengths_nm"]

        out_dir = step08_root / acquisition_id
        out_dir.mkdir(parents=True, exist_ok=True)

        records, arrays = compute_spectral_statistics(spectra, wavelengths)

        save_csv(records, out_dir / "extracted_spectra_statistics.csv")

        plot_mean_p05_p95(
            wavelengths=wavelengths,
            arrays=arrays,
            output_path=out_dir / "spectral_signature_mean_p05_p95.png",
            title=f"{acquisition_id} - Plant spectral signature"
        )

        plot_mean_std(
            wavelengths=wavelengths,
            arrays=arrays,
            output_path=out_dir / "spectral_signature_mean_std.png",
            title=f"{acquisition_id} - Plant spectral signature mean ± SD"
        )

        plot_median_iqr(
            wavelengths=wavelengths,
            arrays=arrays,
            output_path=out_dir / "spectral_signature_median_iqr.png",
            title=f"{acquisition_id} - Plant spectral signature median and IQR"
        )

        summary = {
            "created_at": datetime.now().isoformat(),
            "acquisition_id": acquisition_id,
            "n_pixels": int(spectra.shape[0]),
            "n_bands": int(spectra.shape[1]),
            "first_wavelength_nm": float(wavelengths[0]),
            "last_wavelength_nm": float(wavelengths[-1]),
            "mean_reflectance_global": float(np.nanmean(spectra)),
            "median_reflectance_global": float(np.nanmedian(spectra)),
            "std_reflectance_global": float(np.nanstd(spectra)),
            "outputs": {
                "statistics_csv": str(out_dir / "extracted_spectra_statistics.csv"),
                "mean_p05_p95_png": str(out_dir / "spectral_signature_mean_p05_p95.png"),
                "mean_std_png": str(out_dir / "spectral_signature_mean_std.png"),
                "median_iqr_png": str(out_dir / "spectral_signature_median_iqr.png"),
            }
        }

        save_json(summary, out_dir / "spectral_qc_summary.json")
        global_summary.append(summary)

        print(f"Pixel: {summary['n_pixels']}")
        print(f"Bande: {summary['n_bands']}")
        print(f"Range: {summary['first_wavelength_nm']:.2f} - {summary['last_wavelength_nm']:.2f} nm")
        print(f"Output salvati in: {out_dir}")

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