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
    """
    Esporta la matrice pixel x lunghezze d'onda in CSV compresso.

    Righe = pixel.
    Colonne = pixel_id, row, col, wl_XXXnm...
    Valori = reflectance calibrata originale, estratta solo dai pixel pianta validi.

    Il .csv.gz è compresso lossless ed è leggibile in R/Python.
    """
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

    step04_root = output_root / "04_radiometric_calibration_qc"
    step06_root = output_root / "06_plant_segmentation"
    step07_root = output_root / "07_extracted_plant_pixels"
    step07_root.mkdir(parents=True, exist_ok=True)

    acquisitions = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    global_summary = []

    for acquisition_dir in acquisitions:
        acquisition_id = acquisition_dir.name
        print(f"\nStep 07 - estrazione pixel pianta: {acquisition_id}")

        out_dir = step07_root / acquisition_id
        out_dir.mkdir(parents=True, exist_ok=True)

        hdr_path = sorted((acquisition_dir / "results").glob("REFLECTANCE_*.hdr"))[0]
        dat_path = sorted((acquisition_dir / "results").glob("REFLECTANCE_*.dat"))[0]

        valid_band_indices_path = step04_root / acquisition_id / "valid_band_indices.npy"
        valid_wavelengths_path = step04_root / acquisition_id / "valid_wavelengths_nm.npy"
        radiometric_mask_path = step04_root / acquisition_id / "radiometric_validity_mask.npy"
        plant_mask_path = step06_root / acquisition_id / "plant_mask_clean.npy"

        for p in [
            valid_band_indices_path,
            valid_wavelengths_path,
            radiometric_mask_path,
            plant_mask_path,
        ]:
            if not p.exists():
                raise FileNotFoundError(f"File mancante: {p}")

        valid_band_indices = np.load(valid_band_indices_path)
        valid_wavelengths = np.load(valid_wavelengths_path)
        radiometric_mask = np.load(radiometric_mask_path)
        plant_mask = np.load(plant_mask_path)

        cube, wavelengths, metadata = load_envi_cube(dat_path, hdr_path)

        if cube.shape[:2] != plant_mask.shape:
            raise ValueError(
                f"Shape cubo {cube.shape[:2]} diversa da plant mask {plant_mask.shape}"
            )

        if cube.shape[:2] != radiometric_mask.shape:
            raise ValueError(
                f"Shape cubo {cube.shape[:2]} diversa da radiometric mask {radiometric_mask.shape}"
            )

        final_mask = plant_mask.astype(bool) & radiometric_mask.astype(bool)

        rows, cols = np.where(final_mask)

        if rows.size == 0:
            raise RuntimeError(f"{acquisition_id}: nessun pixel valido da estrarre.")

        subcube = cube[:, :, valid_band_indices]
        spectra = subcube[rows, cols, :]

        finite_spectra_mask = np.all(np.isfinite(spectra), axis=1)
        spectra = spectra[finite_spectra_mask]
        rows = rows[finite_spectra_mask]
        cols = cols[finite_spectra_mask]

        spectra = spectra.astype(np.float32)
        rows = rows.astype(np.int32)
        cols = cols.astype(np.int32)
        valid_wavelengths = valid_wavelengths.astype(np.float32)
        valid_band_indices = valid_band_indices.astype(np.int32)

        np.savez_compressed(
            out_dir / "plant_pixel_spectra.npz",
            spectra=spectra,
            rows=rows,
            cols=cols,
            wavelengths_nm=valid_wavelengths,
            valid_band_indices=valid_band_indices,
        )

        save_pixel_matrix_csv_gz(
            spectra=spectra,
            rows=rows,
            cols=cols,
            wavelengths=valid_wavelengths,
            output_path=out_dir / "plant_pixel_matrix.csv.gz",
        )

        np.save(out_dir / "final_extraction_mask.npy", final_mask)

        save_coordinates_csv(
            rows=rows,
            cols=cols,
            output_path=out_dir / "plant_pixel_coordinates.csv",
        )

        mean_spectrum = np.nanmean(spectra, axis=0)
        std_spectrum = np.nanstd(spectra, axis=0)

        save_mean_spectrum_csv(
            wavelengths=valid_wavelengths,
            mean_spectrum=mean_spectrum,
            std_spectrum=std_spectrum,
            output_path=out_dir / "plant_mean_spectrum.csv",
        )

        summary = {
            "created_at": datetime.now().isoformat(),
            "acquisition_id": acquisition_id,

            "input_reflectance_dat": str(dat_path),
            "input_reflectance_hdr": str(hdr_path),

            "n_total_pixels": int(final_mask.size),
            "n_plant_mask_pixels": int(plant_mask.sum()),
            "n_radiometrically_valid_pixels": int(radiometric_mask.sum()),
            "n_final_extracted_pixels_before_finite_filter": int(final_mask.sum()),
            "n_final_extracted_pixels": int(spectra.shape[0]),

            "n_valid_bands": int(len(valid_band_indices)),
            "first_wavelength_nm": float(valid_wavelengths[0]),
            "last_wavelength_nm": float(valid_wavelengths[-1]),

            "outputs": {
                "plant_pixel_spectra_npz": str(out_dir / "plant_pixel_spectra.npz"),
                "plant_pixel_matrix_csv_gz": str(out_dir / "plant_pixel_matrix.csv.gz"),
                "final_extraction_mask_npy": str(out_dir / "final_extraction_mask.npy"),
                "plant_pixel_coordinates_csv": str(out_dir / "plant_pixel_coordinates.csv"),
                "plant_mean_spectrum_csv": str(out_dir / "plant_mean_spectrum.csv"),
            },

            "note": (
                "Spectra were extracted from the original calibrated REFLECTANCE cube. "
                "Masks were used only to select plant pixels and radiometrically valid pixels. "
                "NPZ is the primary scientific binary format; CSV.GZ is a lossless-compressed "
                "tabular export for R/Python/statistical software."
            )
        }

        save_json(summary, out_dir / "extraction_summary.json")
        global_summary.append(summary)

        print(f"Pixel pianta mask: {summary['n_plant_mask_pixels']}")
        print(f"Pixel finali estratti: {summary['n_final_extracted_pixels']}")
        print(f"Bande estratte: {summary['n_valid_bands']}")
        print(
            f"Range spettrale: "
            f"{summary['first_wavelength_nm']:.2f} - {summary['last_wavelength_nm']:.2f} nm"
        )
        print(f"NPZ: {out_dir / 'plant_pixel_spectra.npz'}")
        print(f"CSV.GZ: {out_dir / 'plant_pixel_matrix.csv.gz'}")
        print(f"Output salvati in: {out_dir}")

    save_json(
        {
            "created_at": datetime.now().isoformat(),
            "n_acquisitions": len(global_summary),
            "summaries": global_summary,
        },
        step07_root / "global_extraction_summary.json",
    )

    print("\nStep 07 completato.")
    print(f"Output salvati in: {step07_root}")


if __name__ == "__main__":
    main()