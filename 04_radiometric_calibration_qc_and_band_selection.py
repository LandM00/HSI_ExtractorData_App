from pathlib import Path
import csv
import json
import yaml
import numpy as np
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
    dat_path = Path(dat_path)
    hdr_path = Path(hdr_path)

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
        raise ValueError(
            f"Dimensione dati non coerente per {dat_path}. "
            f"Letti {arr.size}, attesi {expected_values}"
        )

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
# FILE DISCOVERY
# ============================================================

def find_acquisition_files(acquisition_dir):
    acquisition_dir = Path(acquisition_dir)
    acquisition_id = acquisition_dir.name

    capture_dir = acquisition_dir / "capture"
    results_dir = acquisition_dir / "results"

    dark_raws = sorted(capture_dir.glob("DARKREF_*.raw"))
    dark_hdrs = sorted(capture_dir.glob("DARKREF_*.hdr"))
    white_raws = sorted(capture_dir.glob("WHITEREF_*.raw"))
    white_hdrs = sorted(capture_dir.glob("WHITEREF_*.hdr"))
    reflectance_dats = sorted(results_dir.glob("REFLECTANCE_*.dat"))
    reflectance_hdrs = sorted(results_dir.glob("REFLECTANCE_*.hdr"))

    if not dark_raws or not dark_hdrs:
        raise FileNotFoundError(f"{acquisition_id}: DARKREF mancante")
    if not white_raws or not white_hdrs:
        raise FileNotFoundError(f"{acquisition_id}: WHITEREF mancante")
    if not reflectance_dats or not reflectance_hdrs:
        raise FileNotFoundError(f"{acquisition_id}: REFLECTANCE mancante")

    files = {
        "raw_dat": capture_dir / f"{acquisition_id}.raw",
        "raw_hdr": capture_dir / f"{acquisition_id}.hdr",
        "dark_dat": dark_raws[0],
        "dark_hdr": dark_hdrs[0],
        "white_dat": white_raws[0],
        "white_hdr": white_hdrs[0],
        "reflectance_dat": reflectance_dats[0],
        "reflectance_hdr": reflectance_hdrs[0],
    }

    for key, path in files.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"{acquisition_id}: file mancante {key}: {path}")

    return files


# ============================================================
# CALIBRATION
# ============================================================

def broadcast_reference(ref_cube, target_lines):
    if ref_cube.shape[0] == 1:
        return np.repeat(ref_cube, target_lines, axis=0)
    return ref_cube


def compute_recalculated_reflectance(raw, dark, white, epsilon=1e-6):
    raw_f = raw.astype(np.float32)
    dark_f = dark.astype(np.float32)
    white_f = white.astype(np.float32)

    denominator = white_f - dark_f
    denominator_valid = np.abs(denominator) > epsilon

    recalculated = np.full(raw_f.shape, np.nan, dtype=np.float32)

    recalculated[denominator_valid] = (
        raw_f[denominator_valid] - dark_f[denominator_valid]
    ) / denominator[denominator_valid]

    return recalculated, denominator, denominator_valid


# ============================================================
# QC FUNCTIONS
# ============================================================

def band_qc_report(
    reflectance,
    denominator,
    wavelengths,
    min_reflectance,
    max_reflectance,
    min_pct_valid_reflectance,
    min_denominator_median,
):
    records = []

    lines, samples, bands = reflectance.shape
    n_pixels = lines * samples

    for b in range(bands):
        refl_band = reflectance[:, :, b]
        den_band = denominator[:, :, b]

        wl = float(wavelengths[b])

        finite = np.isfinite(refl_band)
        in_range = (refl_band >= min_reflectance) & (refl_band <= max_reflectance)
        valid_refl = finite & in_range

        pct_valid_refl = 100 * valid_refl.sum() / n_pixels
        pct_finite = 100 * finite.sum() / n_pixels
        pct_negative = 100 * (refl_band < min_reflectance).sum() / n_pixels
        pct_above_max = 100 * (refl_band > max_reflectance).sum() / n_pixels

        den_finite = den_band[np.isfinite(den_band)]

        if den_finite.size:
            den_min = float(np.min(den_finite))
            den_p01 = float(np.percentile(den_finite, 1))
            den_median = float(np.median(den_finite))
            den_p99 = float(np.percentile(den_finite, 99))
            den_max = float(np.max(den_finite))
        else:
            den_min = den_p01 = den_median = den_p99 = den_max = None

        reflectance_ok = pct_valid_refl >= min_pct_valid_reflectance
        denominator_ok = den_median is not None and den_median >= min_denominator_median

        include_band = reflectance_ok and denominator_ok

        reasons = []
        if not reflectance_ok:
            reasons.append("too_many_invalid_reflectance_values")
        if not denominator_ok:
            reasons.append("low_white_minus_dark_denominator")

        finite_values = refl_band[finite]

        if finite_values.size:
            refl_min = float(np.min(finite_values))
            refl_p01 = float(np.percentile(finite_values, 1))
            refl_median = float(np.median(finite_values))
            refl_p99 = float(np.percentile(finite_values, 99))
            refl_max = float(np.max(finite_values))
        else:
            refl_min = refl_p01 = refl_median = refl_p99 = refl_max = None

        records.append({
            "band_index": b,
            "wavelength_nm": wl,

            "pct_finite_reflectance": float(pct_finite),
            "pct_valid_reflectance": float(pct_valid_refl),
            "pct_negative_reflectance": float(pct_negative),
            "pct_above_max_reflectance": float(pct_above_max),

            "reflectance_min": refl_min,
            "reflectance_p01": refl_p01,
            "reflectance_median": refl_median,
            "reflectance_p99": refl_p99,
            "reflectance_max": refl_max,

            "denominator_min": den_min,
            "denominator_p01": den_p01,
            "denominator_median": den_median,
            "denominator_p99": den_p99,
            "denominator_max": den_max,

            "reflectance_ok": reflectance_ok,
            "denominator_ok": denominator_ok,
            "include_band": include_band,
            "exclusion_reasons": ";".join(reasons),
        })

    return records


def compare_software_vs_recalculated(software, recalculated, band_indices, wavelengths):
    records = []

    for b in band_indices:
        sw = software[:, :, b]
        rc = recalculated[:, :, b]

        valid = np.isfinite(sw) & np.isfinite(rc)

        if valid.sum() == 0:
            records.append({
                "band_index": int(b),
                "wavelength_nm": float(wavelengths[b]),
                "n_valid_pixels": 0,
                "median_abs_diff": None,
                "p95_abs_diff": None,
                "mean_abs_diff": None,
                "max_abs_diff": None,
            })
            continue

        diff = sw[valid].astype(np.float64) - rc[valid].astype(np.float64)
        abs_diff = np.abs(diff)

        records.append({
            "band_index": int(b),
            "wavelength_nm": float(wavelengths[b]),
            "n_valid_pixels": int(valid.sum()),
            "median_abs_diff": float(np.median(abs_diff)),
            "p95_abs_diff": float(np.percentile(abs_diff, 95)),
            "mean_abs_diff": float(np.mean(abs_diff)),
            "max_abs_diff": float(np.max(abs_diff)),
        })

    return records


def make_radiometric_validity_mask(
    reflectance,
    valid_band_indices,
    min_reflectance,
    max_reflectance,
):
    subcube = reflectance[:, :, valid_band_indices]

    valid = (
        np.isfinite(subcube)
        & (subcube >= min_reflectance)
        & (subcube <= max_reflectance)
    )

    return np.all(valid, axis=2)


def decide_status(
    n_valid_bands,
    pct_valid_pixels,
    median_band_abs_diff,
    fail_min_valid_bands,
    warning_min_valid_bands,
    fail_min_valid_pixels_pct,
    warning_min_valid_pixels_pct,
    max_allowed_median_abs_diff,
):
    status = "PASS"
    reasons = []

    if n_valid_bands < fail_min_valid_bands:
        status = "FAIL"
        reasons.append("too_few_valid_bands")
    elif n_valid_bands < warning_min_valid_bands:
        status = "WARNING"
        reasons.append("limited_number_of_valid_bands")

    if pct_valid_pixels < fail_min_valid_pixels_pct:
        status = "FAIL"
        reasons.append("too_few_radiometrically_valid_pixels")
    elif pct_valid_pixels < warning_min_valid_pixels_pct and status != "FAIL":
        status = "WARNING"
        reasons.append("limited_number_of_valid_pixels")

    if median_band_abs_diff is not None and median_band_abs_diff > max_allowed_median_abs_diff:
        status = "WARNING" if status != "FAIL" else status
        reasons.append("software_vs_recalculated_reflectance_difference")

    return status, reasons


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
# MAIN
# ============================================================

def main():
    config = load_config()

    dataset_root = Path(config["dataset_root"])
    output_root = Path(config["output_dir"]) / "04_radiometric_calibration_qc"
    output_root.mkdir(parents=True, exist_ok=True)

    qc_cfg = config.get("radiometric_qc", {})

    min_reflectance = qc_cfg.get("min_reflectance", 0.0)
    max_reflectance = qc_cfg.get("max_reflectance", 1.5)

    min_pct_valid_reflectance = qc_cfg.get("min_pct_valid_reflectance", 95.0)
    min_denominator_median = qc_cfg.get("min_white_minus_dark_median_dn", 20.0)

    fail_min_valid_bands = qc_cfg.get("fail_min_valid_bands", 40)
    warning_min_valid_bands = qc_cfg.get("warning_min_valid_bands", 80)

    fail_min_valid_pixels_pct = qc_cfg.get("fail_min_valid_pixels_pct", 50.0)
    warning_min_valid_pixels_pct = qc_cfg.get("warning_min_valid_pixels_pct", 80.0)

    max_allowed_median_abs_diff = qc_cfg.get("max_allowed_median_abs_diff", 1e-5)

    acquisitions = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    global_summaries = []

    for acquisition_dir in acquisitions:
        acquisition_id = acquisition_dir.name
        print(f"\nStep 04 - QC radiometrico senza filtro wavelength: {acquisition_id}")

        out_dir = output_root / acquisition_id
        out_dir.mkdir(parents=True, exist_ok=True)

        files = find_acquisition_files(acquisition_dir)

        raw, wavelengths_raw, raw_meta = load_envi_cube(files["raw_dat"], files["raw_hdr"])
        dark, wavelengths_dark, dark_meta = load_envi_cube(files["dark_dat"], files["dark_hdr"])
        white, wavelengths_white, white_meta = load_envi_cube(files["white_dat"], files["white_hdr"])
        software_reflectance, wavelengths_ref, ref_meta = load_envi_cube(
            files["reflectance_dat"],
            files["reflectance_hdr"],
        )

        if not np.allclose(wavelengths_raw, wavelengths_ref):
            raise RuntimeError(f"{acquisition_id}: wavelengths RAW e REFLECTANCE non coincidono.")

        dark_b = broadcast_reference(dark, raw.shape[0])
        white_b = broadcast_reference(white, raw.shape[0])

        recalculated_reflectance, denominator, denominator_valid = compute_recalculated_reflectance(
            raw=raw,
            dark=dark_b,
            white=white_b,
        )

        band_records = band_qc_report(
            reflectance=software_reflectance,
            denominator=denominator,
            wavelengths=wavelengths_ref,
            min_reflectance=min_reflectance,
            max_reflectance=max_reflectance,
            min_pct_valid_reflectance=min_pct_valid_reflectance,
            min_denominator_median=min_denominator_median,
        )

        valid_bands = [r for r in band_records if r["include_band"]]
        excluded_bands = [r for r in band_records if not r["include_band"]]
        valid_band_indices = [int(r["band_index"]) for r in valid_bands]

        if len(valid_band_indices) == 0:
            raise RuntimeError(f"{acquisition_id}: nessuna banda valida secondo QC radiometrico.")

        validity_mask = make_radiometric_validity_mask(
            reflectance=software_reflectance,
            valid_band_indices=valid_band_indices,
            min_reflectance=min_reflectance,
            max_reflectance=max_reflectance,
        )

        pct_valid_pixels = float(100 * validity_mask.sum() / validity_mask.size)

        comparison_records = compare_software_vs_recalculated(
            software=software_reflectance,
            recalculated=recalculated_reflectance,
            band_indices=valid_band_indices,
            wavelengths=wavelengths_ref,
        )

        valid_diffs = [
            r["median_abs_diff"]
            for r in comparison_records
            if r["median_abs_diff"] is not None
        ]

        median_band_abs_diff = float(np.median(valid_diffs)) if valid_diffs else None
        p95_band_abs_diff = float(np.percentile(valid_diffs, 95)) if valid_diffs else None

        status, status_reasons = decide_status(
            n_valid_bands=len(valid_band_indices),
            pct_valid_pixels=pct_valid_pixels,
            median_band_abs_diff=median_band_abs_diff,
            fail_min_valid_bands=fail_min_valid_bands,
            warning_min_valid_bands=warning_min_valid_bands,
            fail_min_valid_pixels_pct=fail_min_valid_pixels_pct,
            warning_min_valid_pixels_pct=warning_min_valid_pixels_pct,
            max_allowed_median_abs_diff=max_allowed_median_abs_diff,
        )

        save_csv(band_records, out_dir / "band_qc_report.csv")
        save_csv(valid_bands, out_dir / "valid_bands.csv")
        save_csv(excluded_bands, out_dir / "excluded_bands.csv")
        save_csv(comparison_records, out_dir / "software_vs_recalculated_reflectance.csv")

        np.save(out_dir / "radiometric_validity_mask.npy", validity_mask)
        np.save(out_dir / "valid_band_indices.npy", np.array(valid_band_indices, dtype=int))
        np.save(out_dir / "valid_wavelengths_nm.npy", wavelengths_ref[valid_band_indices])

        summary = {
            "created_at": datetime.now().isoformat(),
            "acquisition_id": acquisition_id,
            "status": status,
            "status_reasons": status_reasons,

            "criteria": {
                "wavelength_filter_used": False,
                "min_reflectance": min_reflectance,
                "max_reflectance": max_reflectance,
                "min_pct_valid_reflectance": min_pct_valid_reflectance,
                "min_white_minus_dark_median_dn": min_denominator_median,
                "fail_min_valid_bands": fail_min_valid_bands,
                "warning_min_valid_bands": warning_min_valid_bands,
                "fail_min_valid_pixels_pct": fail_min_valid_pixels_pct,
                "warning_min_valid_pixels_pct": warning_min_valid_pixels_pct,
                "max_allowed_median_abs_diff": max_allowed_median_abs_diff,
            },

            "n_total_bands": int(software_reflectance.shape[2]),
            "n_valid_bands": int(len(valid_band_indices)),
            "n_excluded_bands": int(len(excluded_bands)),
            "first_valid_wavelength_nm": float(wavelengths_ref[valid_band_indices[0]]),
            "last_valid_wavelength_nm": float(wavelengths_ref[valid_band_indices[-1]]),

            "n_pixels": int(validity_mask.size),
            "n_radiometrically_valid_pixels": int(validity_mask.sum()),
            "pct_radiometrically_valid_pixels": pct_valid_pixels,

            "median_band_abs_diff_software_vs_recalculated": median_band_abs_diff,
            "p95_band_abs_diff_software_vs_recalculated": p95_band_abs_diff,

            "input_files": {k: str(v) for k, v in files.items()},
        }

        save_json(summary, out_dir / "radiometric_calibration_qc_summary.json")
        global_summaries.append(summary)

        print(f"Status: {status}")
        print(f"Reason: {status_reasons}")
        print(f"Bande valide QC: {len(valid_band_indices)} / {software_reflectance.shape[2]}")
        print(
            f"Range bande valide QC: "
            f"{wavelengths_ref[valid_band_indices[0]]:.2f} - "
            f"{wavelengths_ref[valid_band_indices[-1]]:.2f} nm"
        )
        print(f"Pixel radiometricamente validi: {pct_valid_pixels:.2f}%")
        print(f"Median abs diff software vs recalculated: {median_band_abs_diff}")

    save_json(
        {
            "created_at": datetime.now().isoformat(),
            "n_acquisitions": len(global_summaries),
            "summaries": global_summaries,
        },
        output_root / "global_radiometric_calibration_qc_summary.json",
    )

    print("\nStep 04 completato.")
    print(f"Output salvati in: {output_root}")


if __name__ == "__main__":
    main()