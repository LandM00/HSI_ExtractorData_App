from pathlib import Path
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
# SUPPORT FUNCTIONS
# ============================================================

def nearest_valid_band(wavelengths, valid_band_indices, target_nm):
    valid_band_indices = np.asarray(valid_band_indices, dtype=int)
    valid_wavelengths = wavelengths[valid_band_indices]
    local_idx = int(np.argmin(np.abs(valid_wavelengths - target_nm)))
    return int(valid_band_indices[local_idx])


def normalize_for_display(img, mask=None, p_low=2, p_high=98):
    img = img.astype(float)

    valid = np.isfinite(img)
    if mask is not None:
        valid = valid & mask

    if valid.sum() == 0:
        return np.zeros_like(img, dtype=float)

    low, high = np.percentile(img[valid], [p_low, p_high])

    if high <= low:
        return np.zeros_like(img, dtype=float)

    out = (img - low) / (high - low)
    return np.clip(out, 0, 1)


def safe_divide(num, den):
    out = np.full(num.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 1e-12)
    out[valid] = num[valid] / den[valid]
    return out


def create_pseudo_rgb(cube, wavelengths, valid_band_indices, validity_mask):
    r_idx = nearest_valid_band(wavelengths, valid_band_indices, 660)
    g_idx = nearest_valid_band(wavelengths, valid_band_indices, 550)
    b_idx = nearest_valid_band(wavelengths, valid_band_indices, 470)

    rgb = np.dstack([
        normalize_for_display(cube[:, :, r_idx], validity_mask),
        normalize_for_display(cube[:, :, g_idx], validity_mask),
        normalize_for_display(cube[:, :, b_idx], validity_mask),
    ])

    return rgb, {
        "red_band_index": r_idx,
        "green_band_index": g_idx,
        "blue_band_index": b_idx,
        "red_wavelength_nm": float(wavelengths[r_idx]),
        "green_wavelength_nm": float(wavelengths[g_idx]),
        "blue_wavelength_nm": float(wavelengths[b_idx]),
    }


def create_false_color(cube, wavelengths, valid_band_indices, validity_mask):
    nir_idx = nearest_valid_band(wavelengths, valid_band_indices, 800)
    red_idx = nearest_valid_band(wavelengths, valid_band_indices, 670)
    green_idx = nearest_valid_band(wavelengths, valid_band_indices, 550)

    false_color = np.dstack([
        normalize_for_display(cube[:, :, nir_idx], validity_mask),
        normalize_for_display(cube[:, :, red_idx], validity_mask),
        normalize_for_display(cube[:, :, green_idx], validity_mask),
    ])

    return false_color, {
        "nir_band_index": nir_idx,
        "red_band_index": red_idx,
        "green_band_index": green_idx,
        "nir_wavelength_nm": float(wavelengths[nir_idx]),
        "red_wavelength_nm": float(wavelengths[red_idx]),
        "green_wavelength_nm": float(wavelengths[green_idx]),
    }


def calculate_ndvi(cube, wavelengths, valid_band_indices):
    red_idx = nearest_valid_band(wavelengths, valid_band_indices, 670)
    nir_idx = nearest_valid_band(wavelengths, valid_band_indices, 800)

    red = cube[:, :, red_idx].astype(np.float32)
    nir = cube[:, :, nir_idx].astype(np.float32)

    ndvi = safe_divide(nir - red, nir + red)

    return ndvi, {
        "red_band_index": red_idx,
        "nir_band_index": nir_idx,
        "red_wavelength_nm": float(wavelengths[red_idx]),
        "nir_wavelength_nm": float(wavelengths[nir_idx]),
    }


def calculate_nir_red_ratio(cube, wavelengths, valid_band_indices):
    red_idx = nearest_valid_band(wavelengths, valid_band_indices, 670)
    nir_idx = nearest_valid_band(wavelengths, valid_band_indices, 800)

    red = cube[:, :, red_idx].astype(np.float32)
    nir = cube[:, :, nir_idx].astype(np.float32)

    ratio = safe_divide(nir, red)

    return ratio, {
        "red_band_index": red_idx,
        "nir_band_index": nir_idx,
        "red_wavelength_nm": float(wavelengths[red_idx]),
        "nir_wavelength_nm": float(wavelengths[nir_idx]),
    }


# ============================================================
# SAVE FIGURES
# ============================================================

def save_rgb_image(img, output_path, title):
    plt.figure(figsize=(7, 7))
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_index_image(index_map, output_path, title, vmin=None, vmax=None, cmap="viridis"):
    plt.figure(figsize=(7, 6))
    plt.imshow(index_map, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


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
    step05_root = output_root / "05_support_images"
    step05_root.mkdir(parents=True, exist_ok=True)

    acquisitions = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    global_summary = []

    for acquisition_dir in acquisitions:
        acquisition_id = acquisition_dir.name
        print(f"\nStep 05 - immagini di supporto: {acquisition_id}")

        qc_summary_path = step04_root / acquisition_id / "radiometric_calibration_qc_summary.json"

        if not qc_summary_path.exists():
            raise FileNotFoundError(
                f"Output Step 04 non trovato per {acquisition_id}: {qc_summary_path}"
            )

        with open(qc_summary_path, "r", encoding="utf-8") as f:
            qc_summary = json.load(f)

        if qc_summary["status"] == "FAIL":
            raise RuntimeError(f"{acquisition_id}: Step 04 in FAIL. Non procedo.")

        valid_band_indices = np.load(step04_root / acquisition_id / "valid_band_indices.npy")
        validity_mask = np.load(step04_root / acquisition_id / "radiometric_validity_mask.npy")

        hdr_path = sorted((acquisition_dir / "results").glob("REFLECTANCE_*.hdr"))[0]
        dat_path = sorted((acquisition_dir / "results").glob("REFLECTANCE_*.dat"))[0]

        cube, wavelengths, metadata = load_envi_cube(dat_path, hdr_path)

        out_dir = step05_root / acquisition_id
        out_dir.mkdir(parents=True, exist_ok=True)

        pseudo_rgb, pseudo_rgb_info = create_pseudo_rgb(
            cube=cube,
            wavelengths=wavelengths,
            valid_band_indices=valid_band_indices,
            validity_mask=validity_mask,
        )

        false_color, false_color_info = create_false_color(
            cube=cube,
            wavelengths=wavelengths,
            valid_band_indices=valid_band_indices,
            validity_mask=validity_mask,
        )

        ndvi, ndvi_info = calculate_ndvi(
            cube=cube,
            wavelengths=wavelengths,
            valid_band_indices=valid_band_indices,
        )

        nir_red_ratio, ratio_info = calculate_nir_red_ratio(
            cube=cube,
            wavelengths=wavelengths,
            valid_band_indices=valid_band_indices,
        )

        # Applica validità radiometrica solo per visualizzazione/segmentazione successiva
        ndvi_masked = ndvi.copy()
        ndvi_masked[~validity_mask] = np.nan

        ratio_masked = nir_red_ratio.copy()
        ratio_masked[~validity_mask] = np.nan

        # Salvataggi numerici
        np.save(out_dir / "pseudo_rgb.npy", pseudo_rgb)
        np.save(out_dir / "false_color_nir_red_green.npy", false_color)
        np.save(out_dir / "ndvi.npy", ndvi)
        np.save(out_dir / "ndvi_masked.npy", ndvi_masked)
        np.save(out_dir / "nir_red_ratio.npy", nir_red_ratio)
        np.save(out_dir / "nir_red_ratio_masked.npy", ratio_masked)

        # Salvataggi visivi
        save_rgb_image(
            pseudo_rgb,
            out_dir / "pseudo_rgb.png",
            f"{acquisition_id} - Pseudo RGB"
        )

        save_rgb_image(
            false_color,
            out_dir / "false_color_nir_red_green.png",
            f"{acquisition_id} - False color NIR/Red/Green"
        )

        save_index_image(
            ndvi_masked,
            out_dir / "ndvi.png",
            f"{acquisition_id} - NDVI",
            vmin=-1,
            vmax=1,
            cmap="RdYlGn"
        )

        save_index_image(
            ratio_masked,
            out_dir / "nir_red_ratio.png",
            f"{acquisition_id} - NIR/Red ratio",
            cmap="viridis"
        )

        summary = {
            "created_at": datetime.now().isoformat(),
            "acquisition_id": acquisition_id,
            "step04_status": qc_summary["status"],
            "n_valid_bands": int(len(valid_band_indices)),
            "validity_mask_valid_pixel_pct": float(100 * validity_mask.sum() / validity_mask.size),
            "pseudo_rgb": pseudo_rgb_info,
            "false_color_nir_red_green": false_color_info,
            "ndvi": ndvi_info,
            "nir_red_ratio": ratio_info,
            "outputs": {
                "pseudo_rgb_png": str(out_dir / "pseudo_rgb.png"),
                "false_color_png": str(out_dir / "false_color_nir_red_green.png"),
                "ndvi_png": str(out_dir / "ndvi.png"),
                "nir_red_ratio_png": str(out_dir / "nir_red_ratio.png"),
                "ndvi_npy": str(out_dir / "ndvi.npy"),
                "ndvi_masked_npy": str(out_dir / "ndvi_masked.npy"),
                "nir_red_ratio_npy": str(out_dir / "nir_red_ratio.npy"),
                "nir_red_ratio_masked_npy": str(out_dir / "nir_red_ratio_masked.npy"),
            }
        }

        save_json(summary, out_dir / "support_images_summary.json")
        global_summary.append(summary)

        print("Creati:")
        print(f"- pseudo_rgb.png")
        print(f"- false_color_nir_red_green.png")
        print(f"- ndvi.png")
        print(f"- nir_red_ratio.png")

    save_json(
        {
            "created_at": datetime.now().isoformat(),
            "n_acquisitions": len(global_summary),
            "summaries": global_summary,
        },
        step05_root / "global_support_images_summary.json"
    )

    print("\nStep 05 completato.")
    print(f"Output salvati in: {step05_root}")


if __name__ == "__main__":
    main()