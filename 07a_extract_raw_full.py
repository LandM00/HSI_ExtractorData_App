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
# ENVI IO
# ============================================================

def parse_envi_header(hdr_path):
    metadata = {}
    current_key = None
    collecting_multiline = False
    multiline_value = []

    with open(hdr_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line or line.upper() == "ENVI":
                continue

            if collecting_multiline:
                multiline_value.append(line)
                if "}" in line:
                    metadata[current_key] = " ".join(multiline_value)
                    collecting_multiline = False
                    current_key = None
                    multiline_value = []
                continue

            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip().lower()
                value = value.strip()

                if value.startswith("{") and not value.endswith("}"):
                    current_key = key
                    collecting_multiline = True
                    multiline_value = [value]
                else:
                    metadata[key] = value

    return metadata


def clean_brace_list(value):
    if value is None:
        return []
    value = value.replace("{", "").replace("}", "")
    return [p.strip() for p in value.replace("\n", " ").split(",") if p.strip()]


def to_float_list(value):
    out = []
    for x in clean_brace_list(value):
        try:
            out.append(float(x))
        except Exception:
            pass
    return out


def to_int(value):
    try:
        return int(str(value).strip())
    except Exception:
        return None


def envi_dtype(data_type):
    mapping = {
        1: np.uint8,
        2: np.int16,
        3: np.int32,
        4: np.float32,
        5: np.float64,
        12: np.uint16,
        13: np.uint32,
        14: np.int64,
        15: np.uint64,
    }

    if data_type not in mapping:
        raise ValueError(f"ENVI data type non supportato: {data_type}")

    return mapping[data_type]


def load_envi_cube(dat_path, hdr_path):
    metadata = parse_envi_header(hdr_path)

    samples = to_int(metadata.get("samples"))
    lines = to_int(metadata.get("lines"))
    bands = to_int(metadata.get("bands"))
    header_offset = to_int(metadata.get("header offset")) or 0
    data_type = to_int(metadata.get("data type"))
    interleave = metadata.get("interleave", "").lower()

    dtype = envi_dtype(data_type)
    expected_values = samples * lines * bands

    arr = np.fromfile(
        dat_path,
        dtype=dtype,
        offset=header_offset,
        count=expected_values,
    )

    if arr.size != expected_values:
        raise ValueError(f"Letti {arr.size} valori, attesi {expected_values}: {dat_path}")

    if interleave == "bil":
        cube = arr.reshape((lines, bands, samples))
        cube = np.transpose(cube, (0, 2, 1))
    elif interleave == "bsq":
        cube = arr.reshape((bands, lines, samples))
        cube = np.transpose(cube, (1, 2, 0))
    elif interleave == "bip":
        cube = arr.reshape((lines, samples, bands))
    else:
        raise ValueError(f"Interleave non supportato: {interleave}")

    wavelengths = np.array(to_float_list(metadata.get("wavelength")), dtype=float)

    return cube, wavelengths, metadata


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
# MAIN
# ============================================================

def main():
    config = load_config()

    dataset_root = Path(config["dataset_root"])
    output_root = Path(config["output_dir"])

    step06_root = output_root / "06_plant_segmentation"
    step07_root = output_root / "07_extracted_plant_pixels"
    step07_root.mkdir(parents=True, exist_ok=True)

    acquisitions = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    global_summary = []

    for acquisition_dir in acquisitions:
        acquisition_id = acquisition_dir.name
        print(f"\nStep 07a - estrazione RAW_FULL pixel pianta: {acquisition_id}")

        out_dir = step07_root / acquisition_id
        out_dir.mkdir(parents=True, exist_ok=True)

        hdr_path = sorted((acquisition_dir / "results").glob("REFLECTANCE_*.hdr"))[0]
        dat_path = sorted((acquisition_dir / "results").glob("REFLECTANCE_*.dat"))[0]

        plant_mask_path = step06_root / acquisition_id / "plant_mask_clean.npy"

        if not plant_mask_path.exists():
            raise FileNotFoundError(f"File mancante: {plant_mask_path}")

        plant_mask = np.load(plant_mask_path)

        cube, wavelengths, metadata = load_envi_cube(dat_path, hdr_path)

        if cube.shape[:2] != plant_mask.shape:
            raise ValueError(
                f"Shape cubo {cube.shape[:2]} diversa da plant mask {plant_mask.shape}"
            )

        final_mask = plant_mask.astype(bool)
        rows, cols = np.where(final_mask)

        if rows.size == 0:
            raise RuntimeError(f"{acquisition_id}: nessun pixel pianta da estrarre.")

        spectra_raw_full = cube[rows, cols, :].astype(np.float32)

        rows = rows.astype(np.int32)
        cols = cols.astype(np.int32)
        wavelengths = wavelengths.astype(np.float32)

        n_pixels = int(spectra_raw_full.shape[0])
        n_bands = int(spectra_raw_full.shape[1])
        n_values = int(spectra_raw_full.size)

        n_finite_values = int(np.isfinite(spectra_raw_full).sum())
        n_nan_values = int(np.isnan(spectra_raw_full).sum())
        n_inf_values = int(np.isinf(spectra_raw_full).sum())

        pct_finite_values = float(100 * n_finite_values / n_values)
        pct_nan_values = float(100 * n_nan_values / n_values)
        pct_inf_values = float(100 * n_inf_values / n_values)

        # ------------------------------------------------------------
        # Save RAW_FULL outputs
        # ------------------------------------------------------------
        np.savez_compressed(
            out_dir / "plant_pixel_spectra_RAW_FULL.npz",
            spectra=spectra_raw_full,
            rows=rows,
            cols=cols,
            wavelengths_nm=wavelengths,
        )

        save_pixel_matrix_csv_gz(
            spectra=spectra_raw_full,
            rows=rows,
            cols=cols,
            wavelengths=wavelengths,
            output_path=out_dir / "plant_pixel_matrix_RAW_FULL.csv.gz",
        )

        save_coordinates_csv(
            rows=rows,
            cols=cols,
            output_path=out_dir / "plant_pixel_coordinates_RAW_FULL.csv",
        )

        np.save(out_dir / "final_plant_mask.npy", final_mask)

        mean_spectrum_raw = np.nanmean(spectra_raw_full, axis=0)
        std_spectrum_raw = np.nanstd(spectra_raw_full, axis=0)

        save_mean_spectrum_csv(
            wavelengths=wavelengths,
            mean_spectrum=mean_spectrum_raw,
            std_spectrum=std_spectrum_raw,
            output_path=out_dir / "plant_mean_spectrum_RAW_FULL.csv",
        )

        summary = {
            "created_at": datetime.now().isoformat(),
            "acquisition_id": acquisition_id,

            "input_reflectance_dat": str(dat_path),
            "input_reflectance_hdr": str(hdr_path),
            "input_plant_mask": str(plant_mask_path),

            "n_total_image_pixels": int(final_mask.size),
            "n_plant_pixels": n_pixels,
            "pct_plant_pixels": float(100 * n_pixels / final_mask.size),

            "n_original_bands": n_bands,
            "first_wavelength_nm": float(wavelengths[0]),
            "last_wavelength_nm": float(wavelengths[-1]),

            "raw_full": {
                "description": (
                    "All plant pixels and all original spectral bands. "
                    "No wavelength filtering and no value cleaning applied."
                ),
                "n_pixels": n_pixels,
                "n_bands": n_bands,
                "n_values": n_values,
                "n_finite_values": n_finite_values,
                "n_nan_values": n_nan_values,
                "n_inf_values": n_inf_values,
                "pct_finite_values": pct_finite_values,
                "pct_nan_values": pct_nan_values,
                "pct_inf_values": pct_inf_values,
            },

            "outputs": {
                "plant_pixel_spectra_RAW_FULL_npz": str(out_dir / "plant_pixel_spectra_RAW_FULL.npz"),
                "plant_pixel_matrix_RAW_FULL_csv_gz": str(out_dir / "plant_pixel_matrix_RAW_FULL.csv.gz"),
                "plant_pixel_coordinates_RAW_FULL_csv": str(out_dir / "plant_pixel_coordinates_RAW_FULL.csv"),
                "plant_mean_spectrum_RAW_FULL_csv": str(out_dir / "plant_mean_spectrum_RAW_FULL.csv"),
                "final_plant_mask_npy": str(out_dir / "final_plant_mask.npy"),
            },

            "note": (
                "This step extracts RAW_FULL spectra only. CLEAN datasets are generated "
                "in the next step from RAW_FULL, using thresholds defined in config.yaml."
            )
        }

        save_json(summary, out_dir / "extraction_raw_full_summary.json")
        global_summary.append(summary)

        print(f"Pixel pianta RAW_FULL: {n_pixels}")
        print(f"Bande originali salvate: {n_bands}")
        print(f"Range spettrale: {wavelengths[0]:.2f} - {wavelengths[-1]:.2f} nm")
        print(f"Valori finiti: {pct_finite_values:.4f}%")
        print(f"RAW_FULL NPZ: {out_dir / 'plant_pixel_spectra_RAW_FULL.npz'}")
        print(f"RAW_FULL CSV.GZ: {out_dir / 'plant_pixel_matrix_RAW_FULL.csv.gz'}")
        print(f"Output salvati in: {out_dir}")

    save_json(
        {
            "created_at": datetime.now().isoformat(),
            "n_acquisitions": len(global_summary),
            "summaries": global_summary,
        },
        step07_root / "global_extraction_raw_full_summary.json",
    )

    print("\nStep 07a completato.")
    print(f"Output salvati in: {step07_root}")


if __name__ == "__main__":
    main()