from pathlib import Path
import json
import csv
import yaml
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_envi_header(hdr_path):
    hdr_path = Path(hdr_path)

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
    parts = [p.strip() for p in value.replace("\n", " ").split(",")]
    return [p for p in parts if p != ""]


def to_float_list(value):
    out = []
    for item in clean_brace_list(value):
        try:
            out.append(float(item))
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
        count=expected_values
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


def qc_band_statistics(cube, wavelengths):
    records = []

    n_lines, n_samples, n_bands = cube.shape
    n_pixels = n_lines * n_samples

    for b in range(n_bands):
        band = cube[:, :, b]

        finite_mask = np.isfinite(band)
        finite_values = band[finite_mask]

        n_finite = int(finite_mask.sum())
        n_nan = int(np.isnan(band).sum())
        n_inf = int(np.isinf(band).sum())

        n_negative = int((band < 0).sum())
        n_gt_1 = int((band > 1).sum())
        n_gt_1_5 = int((band > 1.5).sum())
        n_gt_10 = int((band > 10).sum())
        n_gt_100 = int((band > 100).sum())

        if finite_values.size > 0:
            min_val = float(np.min(finite_values))
            max_val = float(np.max(finite_values))
            mean_val = float(np.mean(finite_values))
            median_val = float(np.median(finite_values))
            p01 = float(np.percentile(finite_values, 1))
            p05 = float(np.percentile(finite_values, 5))
            p50 = float(np.percentile(finite_values, 50))
            p95 = float(np.percentile(finite_values, 95))
            p99 = float(np.percentile(finite_values, 99))
        else:
            min_val = max_val = mean_val = median_val = None
            p01 = p05 = p50 = p95 = p99 = None

        records.append({
            "band_index": b,
            "wavelength_nm": float(wavelengths[b]) if b < len(wavelengths) else None,

            "n_pixels": n_pixels,
            "n_finite": n_finite,
            "n_nan": n_nan,
            "n_inf": n_inf,

            "pct_finite": 100 * n_finite / n_pixels,
            "pct_nan": 100 * n_nan / n_pixels,
            "pct_inf": 100 * n_inf / n_pixels,

            "n_negative": n_negative,
            "n_gt_1": n_gt_1,
            "n_gt_1_5": n_gt_1_5,
            "n_gt_10": n_gt_10,
            "n_gt_100": n_gt_100,

            "pct_negative": 100 * n_negative / n_pixels,
            "pct_gt_1": 100 * n_gt_1 / n_pixels,
            "pct_gt_1_5": 100 * n_gt_1_5 / n_pixels,
            "pct_gt_10": 100 * n_gt_10 / n_pixels,
            "pct_gt_100": 100 * n_gt_100 / n_pixels,

            "min": min_val,
            "p01": p01,
            "p05": p05,
            "p50": p50,
            "mean": mean_val,
            "median": median_val,
            "p95": p95,
            "p99": p99,
            "max": max_val,
        })

    return records


def qc_cube_summary(cube, wavelengths, metadata, dat_path, hdr_path):
    finite_mask = np.isfinite(cube)
    finite_values = cube[finite_mask]

    n_total = int(cube.size)
    n_finite = int(finite_mask.sum())

    n_negative = int((cube < 0).sum())
    n_gt_1 = int((cube > 1).sum())
    n_gt_1_5 = int((cube > 1.5).sum())
    n_gt_10 = int((cube > 10).sum())
    n_gt_100 = int((cube > 100).sum())

    summary = {
        "dat_path": str(dat_path),
        "hdr_path": str(hdr_path),
        "shape": list(cube.shape),
        "lines": int(cube.shape[0]),
        "samples": int(cube.shape[1]),
        "bands": int(cube.shape[2]),

        "n_total_values": n_total,
        "n_finite_values": n_finite,
        "pct_finite_values": float(100 * n_finite / n_total),

        "n_nan_values": int(np.isnan(cube).sum()),
        "n_inf_values": int(np.isinf(cube).sum()),

        "n_negative_values": n_negative,
        "n_gt_1_values": n_gt_1,
        "n_gt_1_5_values": n_gt_1_5,
        "n_gt_10_values": n_gt_10,
        "n_gt_100_values": n_gt_100,

        "pct_negative_values": float(100 * n_negative / n_total),
        "pct_gt_1_values": float(100 * n_gt_1 / n_total),
        "pct_gt_1_5_values": float(100 * n_gt_1_5 / n_total),
        "pct_gt_10_values": float(100 * n_gt_10 / n_total),
        "pct_gt_100_values": float(100 * n_gt_100 / n_total),

        "wavelength_min_nm": float(np.min(wavelengths)),
        "wavelength_max_nm": float(np.max(wavelengths)),
        "n_wavelengths": int(len(wavelengths)),
        "interleave": metadata.get("interleave"),
        "data_type": metadata.get("data type"),
        "wavelength_units": metadata.get("wavelength units"),
    }

    if finite_values.size > 0:
        summary.update({
            "global_min": float(np.min(finite_values)),
            "global_p01": float(np.percentile(finite_values, 1)),
            "global_p05": float(np.percentile(finite_values, 5)),
            "global_p50": float(np.percentile(finite_values, 50)),
            "global_median": float(np.median(finite_values)),
            "global_mean": float(np.mean(finite_values)),
            "global_p95": float(np.percentile(finite_values, 95)),
            "global_p99": float(np.percentile(finite_values, 99)),
            "global_max": float(np.max(finite_values)),
        })

    return summary


def save_reflectance_histogram(cube, summary, output_path, acquisition_id):
    """
    Salva un istogramma robusto della distribuzione della reflectance.
    Il range è limitato a 0–2.5 per visualizzare la distribuzione reale,
    senza farsi dominare dagli outlier estremi tipo ±1e38.
    """
    finite_values = cube[np.isfinite(cube)]

    if finite_values.size == 0:
        return

    finite_values = finite_values.astype(np.float64)

    plt.figure(figsize=(9, 5))

    plt.hist(
        finite_values,
        bins=300,
        range=(0, 2.5),
        log=True
    )

    if summary.get("global_p50") is not None:
        plt.axvline(summary["global_p50"], linestyle="--", label="p50 / median")

    if summary.get("global_p95") is not None:
        plt.axvline(summary["global_p95"], linestyle="--", label="p95")

    if summary.get("global_p99") is not None:
        plt.axvline(summary["global_p99"], linestyle="--", label="p99")

    plt.axvline(1.0, linestyle=":", label="Reflectance = 1")
    plt.axvline(1.5, linestyle=":", label="QC threshold = 1.5")

    plt.xlabel("Reflectance")
    plt.ylabel("Count (log scale)")
    plt.title(f"{acquisition_id} - Reflectance distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_csv(records, output_path):
    keys = list(records[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def save_json(payload, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    config = load_config()

    dataset_root = Path(config["dataset_root"])
    output_root = Path(config["output_dir"]) / "03_reflectance_validation"
    output_root.mkdir(parents=True, exist_ok=True)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root non trovata: {dataset_root}")

    acquisitions = sorted([p for p in dataset_root.iterdir() if p.is_dir()])

    global_summaries = []

    for acquisition_dir in acquisitions:
        acquisition_id = acquisition_dir.name

        reflectance_hdrs = sorted((acquisition_dir / "results").glob("REFLECTANCE_*.hdr"))
        reflectance_dats = sorted((acquisition_dir / "results").glob("REFLECTANCE_*.dat"))

        if len(reflectance_hdrs) != 1 or len(reflectance_dats) != 1:
            raise RuntimeError(
                f"{acquisition_id}: atteso 1 REFLECTANCE hdr e 1 dat, "
                f"trovati {len(reflectance_hdrs)} hdr e {len(reflectance_dats)} dat"
            )

        hdr_path = reflectance_hdrs[0]
        dat_path = reflectance_dats[0]

        print(f"\nCarico reflectance: {acquisition_id}")

        cube, wavelengths, metadata = load_envi_cube(dat_path, hdr_path)

        print(f"Shape cubo: {cube.shape}")
        print(f"Wavelength range: {wavelengths.min():.2f} - {wavelengths.max():.2f} nm")

        band_records = qc_band_statistics(cube, wavelengths)
        summary = qc_cube_summary(cube, wavelengths, metadata, dat_path, hdr_path)
        summary["acquisition_id"] = acquisition_id
        summary["created_at"] = datetime.now().isoformat()

        acquisition_output = output_root / acquisition_id
        acquisition_output.mkdir(parents=True, exist_ok=True)

        save_csv(
            band_records,
            acquisition_output / "reflectance_band_qc.csv"
        )

        save_json(
            summary,
            acquisition_output / "reflectance_cube_summary.json"
        )

        save_reflectance_histogram(
            cube=cube,
            summary=summary,
            output_path=acquisition_output / "reflectance_distribution_histogram.png",
            acquisition_id=acquisition_id
        )

        global_summaries.append(summary)

        print("QC salvato.")
        print(f"Finite values: {summary['pct_finite_values']:.3f}%")
        print(f"Global min/max: {summary.get('global_min')} / {summary.get('global_max')}")
        print(
            "Global percentili: "
            f"p01={summary.get('global_p01')}, "
            f"p05={summary.get('global_p05')}, "
            f"p50={summary.get('global_p50')}, "
            f"p95={summary.get('global_p95')}, "
            f"p99={summary.get('global_p99')}"
        )
        print(f"Valori < 0: {summary['n_negative_values']} ({summary['pct_negative_values']:.4f}%)")
        print(f"Valori > 1: {summary['n_gt_1_values']} ({summary['pct_gt_1_values']:.4f}%)")
        print(f"Valori > 1.5: {summary['n_gt_1_5_values']} ({summary['pct_gt_1_5_values']:.4f}%)")
        print(f"Valori > 10: {summary['n_gt_10_values']} ({summary['pct_gt_10_values']:.6f}%)")
        print(f"Valori > 100: {summary['n_gt_100_values']} ({summary['pct_gt_100_values']:.6f}%)")
        print(f"Istogramma salvato in: {acquisition_output / 'reflectance_distribution_histogram.png'}")

    save_json(
        {
            "created_at": datetime.now().isoformat(),
            "n_acquisitions": len(global_summaries),
            "summaries": global_summaries,
        },
        output_root / "global_reflectance_summary.json"
    )

    print("\nStep 3 completato.")
    print(f"Output salvati in: {output_root}")


if __name__ == "__main__":
    main()