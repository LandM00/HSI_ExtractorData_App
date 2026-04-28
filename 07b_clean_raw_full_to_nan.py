from pathlib import Path
import csv
import json
import yaml
import numpy as np
import pandas as pd
from datetime import datetime


# ============================================================
# CONFIG
# ============================================================

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# EXPORT
# ============================================================

def save_coordinates_csv(rows, cols, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "col"])
        for r, c in zip(rows, cols):
            writer.writerow([int(r), int(c)])


def save_mean_spectrum_csv(wavelengths, mean_spectrum, std_spectrum, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["wavelength_nm", "mean_reflectance", "std_reflectance"])
        for wl, mean, std in zip(wavelengths, mean_spectrum, std_spectrum):
            writer.writerow([float(wl), float(mean), float(std)])


def save_pixel_matrix_csv_gz(spectra, rows, cols, wavelengths, output_path):
    df = pd.DataFrame(
        spectra.astype(np.float32),
        columns=[f"wl_{float(wl):.2f}nm" for wl in wavelengths]
    )

    df.insert(0, "col", cols.astype(np.int32))
    df.insert(0, "row", rows.astype(np.int32))
    df.insert(0, "pixel_id", np.arange(len(df), dtype=np.int64))

    df.to_csv(output_path, index=False, compression="gzip")


def save_json(payload, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ============================================================
# CLEANING
# ============================================================

def build_clean_nan_dataset(spectra_raw_full, min_value=0.0, max_value=1.5):
    spectra_clean_nan = spectra_raw_full.copy().astype(np.float32)

    invalid_value_mask = (
        ~np.isfinite(spectra_clean_nan) |
        (spectra_clean_nan < min_value) |
        (spectra_clean_nan > max_value)
    )

    spectra_clean_nan[invalid_value_mask] = np.nan

    return spectra_clean_nan, invalid_value_mask


# ============================================================
# MAIN
# ============================================================

def main():
    config = load_config()

    dataset_root = Path(config["dataset_root"])
    output_root = Path(config["output_dir"])

    step07_root = output_root / "07_extracted_plant_pixels"

    clean_cfg = config.get("cleaning", {})
    clean_min_reflectance = float(clean_cfg.get("min_reflectance", 0.0))
    clean_max_reflectance = float(clean_cfg.get("max_reflectance", 1.5))
    clean_method = clean_cfg.get("method", "value_range_to_nan")

    acquisitions = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    global_summary = []

    for acquisition_dir in acquisitions:
        acquisition_id = acquisition_dir.name
        print(f"\nStep 07b - creazione CLEAN_NAN da RAW_FULL: {acquisition_id}")

        out_dir = step07_root / acquisition_id
        out_dir.mkdir(parents=True, exist_ok=True)

        raw_npz = out_dir / "plant_pixel_spectra_RAW_FULL.npz"

        if not raw_npz.exists():
            raise FileNotFoundError(
                f"File RAW_FULL mancante: {raw_npz}. "
                "Esegui prima 07a_extract_raw_full.py"
            )

        data = np.load(raw_npz)

        spectra_raw_full = data["spectra"].astype(np.float32)
        rows = data["rows"].astype(np.int32)
        cols = data["cols"].astype(np.int32)
        wavelengths = data["wavelengths_nm"].astype(np.float32)

        spectra_clean_nan, invalid_value_mask = build_clean_nan_dataset(
            spectra_raw_full=spectra_raw_full,
            min_value=clean_min_reflectance,
            max_value=clean_max_reflectance,
        )

        n_pixels = int(spectra_raw_full.shape[0])
        n_bands = int(spectra_raw_full.shape[1])
        n_total_values = int(spectra_raw_full.size)

        n_invalid_values = int(invalid_value_mask.sum())
        pct_invalid_values = float(100 * n_invalid_values / n_total_values)

        n_pixels_with_at_least_one_invalid = int(np.any(invalid_value_mask, axis=1).sum())
        pct_pixels_with_at_least_one_invalid = float(
            100 * n_pixels_with_at_least_one_invalid / n_pixels
        )

        n_nan_clean = int(np.isnan(spectra_clean_nan).sum())
        pct_nan_clean = float(100 * n_nan_clean / spectra_clean_nan.size)

        n_finite_clean = int(np.isfinite(spectra_clean_nan).sum())
        pct_finite_clean = float(100 * n_finite_clean / spectra_clean_nan.size)

        # ------------------------------------------------------------
        # Save CLEAN_NAN outputs
        # ------------------------------------------------------------
        np.savez_compressed(
            out_dir / "plant_pixel_spectra_CLEAN_NAN.npz",
            spectra=spectra_clean_nan,
            rows=rows,
            cols=cols,
            wavelengths_nm=wavelengths,
            cleaning_min_reflectance=np.array([clean_min_reflectance], dtype=np.float32),
            cleaning_max_reflectance=np.array([clean_max_reflectance], dtype=np.float32),
        )

        save_pixel_matrix_csv_gz(
            spectra=spectra_clean_nan,
            rows=rows,
            cols=cols,
            wavelengths=wavelengths,
            output_path=out_dir / "plant_pixel_matrix_CLEAN_NAN.csv.gz",
        )

        save_coordinates_csv(
            rows=rows,
            cols=cols,
            output_path=out_dir / "plant_pixel_coordinates_CLEAN_NAN.csv",
        )

        np.save(out_dir / "invalid_value_mask_CLEAN_NAN.npy", invalid_value_mask)

        mean_spectrum_clean_nan = np.nanmean(spectra_clean_nan, axis=0)
        std_spectrum_clean_nan = np.nanstd(spectra_clean_nan, axis=0)

        save_mean_spectrum_csv(
            wavelengths=wavelengths,
            mean_spectrum=mean_spectrum_clean_nan,
            std_spectrum=std_spectrum_clean_nan,
            output_path=out_dir / "plant_mean_spectrum_CLEAN_NAN.csv",
        )

        # ------------------------------------------------------------
        # Backward-compatible aliases point to CLEAN_NAN
        # ------------------------------------------------------------
        np.savez_compressed(
            out_dir / "plant_pixel_spectra.npz",
            spectra=spectra_clean_nan,
            rows=rows,
            cols=cols,
            wavelengths_nm=wavelengths,
        )

        save_pixel_matrix_csv_gz(
            spectra=spectra_clean_nan,
            rows=rows,
            cols=cols,
            wavelengths=wavelengths,
            output_path=out_dir / "plant_pixel_matrix.csv.gz",
        )

        save_mean_spectrum_csv(
            wavelengths=wavelengths,
            mean_spectrum=mean_spectrum_clean_nan,
            std_spectrum=std_spectrum_clean_nan,
            output_path=out_dir / "plant_mean_spectrum.csv",
        )

        summary = {
            "created_at": datetime.now().isoformat(),
            "acquisition_id": acquisition_id,

            "input_raw_full_npz": str(raw_npz),

            "clean_nan": {
                "description": (
                    "Same pixels and same original spectral bands as RAW_FULL. "
                    "Non-finite values and values outside the configured reflectance "
                    "range are replaced with NaN."
                ),
                "method": clean_method,
                "min_reflectance": clean_min_reflectance,
                "max_reflectance": clean_max_reflectance,

                "n_pixels": n_pixels,
                "n_bands": n_bands,
                "n_total_values": n_total_values,

                "n_values_set_to_nan": n_invalid_values,
                "pct_values_set_to_nan": pct_invalid_values,

                "n_nan_values": n_nan_clean,
                "pct_nan_values": pct_nan_clean,

                "n_finite_values": n_finite_clean,
                "pct_finite_values": pct_finite_clean,

                "n_pixels_with_at_least_one_invalid_value": n_pixels_with_at_least_one_invalid,
                "pct_pixels_with_at_least_one_invalid_value": pct_pixels_with_at_least_one_invalid,
            },

            "outputs": {
                "plant_pixel_spectra_CLEAN_NAN_npz": str(out_dir / "plant_pixel_spectra_CLEAN_NAN.npz"),
                "plant_pixel_matrix_CLEAN_NAN_csv_gz": str(out_dir / "plant_pixel_matrix_CLEAN_NAN.csv.gz"),
                "plant_pixel_coordinates_CLEAN_NAN_csv": str(out_dir / "plant_pixel_coordinates_CLEAN_NAN.csv"),

                "plant_pixel_spectra_default_npz": str(out_dir / "plant_pixel_spectra.npz"),
                "plant_pixel_matrix_default_csv_gz": str(out_dir / "plant_pixel_matrix.csv.gz"),

                "invalid_value_mask_CLEAN_NAN_npy": str(out_dir / "invalid_value_mask_CLEAN_NAN.npy"),
                "plant_mean_spectrum_CLEAN_NAN_csv": str(out_dir / "plant_mean_spectrum_CLEAN_NAN.csv"),
                "plant_mean_spectrum_default_csv": str(out_dir / "plant_mean_spectrum.csv"),
            },

            "note": (
                "CLEAN_NAN is derived from RAW_FULL. RAW_FULL is not modified. "
                "Default output filenames point to CLEAN_NAN for compatibility with downstream steps."
            )
        }

        save_json(summary, out_dir / "clean_nan_summary.json")
        global_summary.append(summary)

        print(f"Pixel: {n_pixels}")
        print(f"Bande: {n_bands}")
        print(f"Range spettrale: {wavelengths[0]:.2f} - {wavelengths[-1]:.2f} nm")
        print(
            f"Valori impostati a NaN: "
            f"{n_invalid_values} / {n_total_values} ({pct_invalid_values:.4f}%)"
        )
        print(
            f"Pixel con almeno un valore fuori soglia: "
            f"{n_pixels_with_at_least_one_invalid} / {n_pixels} "
            f"({pct_pixels_with_at_least_one_invalid:.2f}%)"
        )
        print(f"CLEAN_NAN NPZ: {out_dir / 'plant_pixel_spectra_CLEAN_NAN.npz'}")
        print(f"CLEAN_NAN CSV.GZ: {out_dir / 'plant_pixel_matrix_CLEAN_NAN.csv.gz'}")
        print(f"Output salvati in: {out_dir}")

    save_json(
        {
            "created_at": datetime.now().isoformat(),
            "n_acquisitions": len(global_summary),
            "summaries": global_summary,
        },
        step07_root / "global_clean_nan_summary.json",
    )

    print("\nStep 07b completato.")
    print(f"Output salvati in: {step07_root}")


if __name__ == "__main__":
    main()