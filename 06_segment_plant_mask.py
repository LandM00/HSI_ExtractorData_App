from pathlib import Path
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from skimage.filters import threshold_otsu
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_opening,
    binary_closing,
    binary_erosion,
    disk,
)
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops


# ============================================================
# CONFIG
# ============================================================

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# UTILS
# ============================================================

def robust_otsu_threshold(values, min_threshold=0.20, max_threshold=0.75):
    values = np.asarray(values)
    values = values[np.isfinite(values)]

    if values.size == 0:
        raise RuntimeError("Nessun valore valido per calcolare Otsu.")

    otsu = float(threshold_otsu(values))
    threshold = float(np.clip(otsu, min_threshold, max_threshold))

    return threshold, otsu


def normalize_01(x):
    x = x.astype(np.float32)
    valid = np.isfinite(x)

    out = np.zeros_like(x, dtype=np.float32)

    if valid.sum() == 0:
        return out

    p2 = np.nanpercentile(x[valid], 2)
    p98 = np.nanpercentile(x[valid], 98)

    if p98 <= p2:
        return out

    out = (x - p2) / (p98 - p2)
    out = np.clip(out, 0, 1)
    out[~valid] = 0

    return out


def compute_exg_from_rgb(pseudo_rgb):
    """
    Excess Green index:
    ExG = 2G - R - B
    utile per distinguere vegetazione verde da sfondo/white reference.
    """
    rgb = np.asarray(pseudo_rgb, dtype=np.float32)

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    exg = 2.0 * g - r - b
    return exg


def create_initial_mask(
    ndvi_masked,
    pseudo_rgb,
    validity_mask,
    ndvi_threshold,
    use_exg=True,
    exg_min_threshold=0.05,
    use_green_dominance=True,
    green_margin=0.02,
):
    """
    Maschera iniziale più robusta:
    - NDVI sopra soglia
    - pixel radiometricamente valido
    - opzionale ExG positivo
    - opzionale green dominance: G > R + margin e G > B + margin
    """
    base = (
        np.isfinite(ndvi_masked)
        & validity_mask.astype(bool)
        & (ndvi_masked >= ndvi_threshold)
    )

    rgb = np.asarray(pseudo_rgb, dtype=np.float32)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    if use_exg:
        exg = compute_exg_from_rgb(rgb)
        exg_norm = normalize_01(exg)
        base = base & (exg_norm >= exg_min_threshold)

    if use_green_dominance:
        base = base & (g >= r + green_margin) & (g >= b + green_margin)

    return base.astype(bool)


def keep_largest_components(mask, max_components=2, min_area_px=300):
    """
    Mantiene le N componenti più grandi.
    Utile per togliere tablet, rumore, oggetti secondari.
    """
    if max_components is None or max_components <= 0:
        return mask.astype(bool)

    lab = label(mask)
    props = regionprops(lab)

    props = [p for p in props if p.area >= min_area_px]

    if not props:
        return mask.astype(bool)

    props_sorted = sorted(props, key=lambda p: p.area, reverse=True)
    keep_labels = [p.label for p in props_sorted[:max_components]]

    out = np.isin(lab, keep_labels)

    return out.astype(bool)


def clean_plant_mask(
    mask,
    min_object_size_px=300,
    min_hole_size_px=300,
    opening_radius_px=1,
    closing_radius_px=2,
    erosion_radius_px=0,
    remove_border_objects=False,
    keep_largest_n_components=2,
):
    clean = mask.astype(bool)

    if remove_border_objects:
        clean = clear_border(clean)

    clean = remove_small_objects(clean, min_size=min_object_size_px)

    if closing_radius_px and closing_radius_px > 0:
        clean = binary_closing(clean, footprint=disk(closing_radius_px))

    if opening_radius_px and opening_radius_px > 0:
        clean = binary_opening(clean, footprint=disk(opening_radius_px))

    clean = remove_small_holes(clean, area_threshold=min_hole_size_px)

    if erosion_radius_px and erosion_radius_px > 0:
        clean = binary_erosion(clean, footprint=disk(erosion_radius_px))

    clean = remove_small_objects(clean, min_size=min_object_size_px)

    clean = keep_largest_components(
        clean,
        max_components=keep_largest_n_components,
        min_area_px=min_object_size_px,
    )

    return clean.astype(bool)


def save_mask_png(mask, output_path, title):
    plt.figure(figsize=(7, 7))
    plt.imshow(mask, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_overlay_png(background_rgb, mask, output_path, title):
    bg = np.asarray(background_rgb, dtype=np.float32).copy()
    bg = np.clip(bg, 0, 1)

    overlay = bg.copy()
    overlay[mask, 0] = 1.0
    overlay[mask, 1] = overlay[mask, 1] * 0.3
    overlay[mask, 2] = overlay[mask, 2] * 0.3

    alpha = 0.45
    combined = bg * (1 - alpha) + overlay * alpha

    plt.figure(figsize=(7, 7))
    plt.imshow(combined)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_ndvi_threshold_png(ndvi_masked, threshold, output_path, title):
    plt.figure(figsize=(7, 6))
    plt.imshow(ndvi_masked, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(label="NDVI")
    plt.contour(ndvi_masked >= threshold, colors="black", linewidths=0.4)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_exg_png(pseudo_rgb, output_path, title):
    exg = compute_exg_from_rgb(pseudo_rgb)
    exg_norm = normalize_01(exg)

    plt.figure(figsize=(7, 6))
    plt.imshow(exg_norm, cmap="Greens", vmin=0, vmax=1)
    plt.colorbar(label="Normalized ExG")
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
    step06_root = output_root / "06_plant_segmentation"
    step06_root.mkdir(parents=True, exist_ok=True)

    seg_cfg = config.get("segmentation", {})

    method = seg_cfg.get("method", "ndvi_exg_otsu")

    ndvi_min_threshold = float(seg_cfg.get("ndvi_min_threshold", 0.20))
    ndvi_max_threshold = float(seg_cfg.get("ndvi_max_threshold", 0.75))

    use_exg = bool(seg_cfg.get("use_exg", True))
    exg_min_threshold = float(seg_cfg.get("exg_min_threshold", 0.05))

    use_green_dominance = bool(seg_cfg.get("use_green_dominance", True))
    green_margin = float(seg_cfg.get("green_margin", 0.02))

    min_object_size_px = int(seg_cfg.get("min_object_size_px", 300))
    min_hole_size_px = int(seg_cfg.get("min_hole_size_px", 300))

    opening_radius_px = int(seg_cfg.get("opening_radius_px", 1))
    closing_radius_px = int(seg_cfg.get("closing_radius_px", 2))
    erosion_radius_px = int(seg_cfg.get("erosion_radius_px", 0))

    remove_border_objects = bool(seg_cfg.get("remove_border_objects", False))
    keep_largest_n_components = int(seg_cfg.get("keep_largest_n_components", 2))

    if method not in ["ndvi_otsu", "ndvi_exg_otsu"]:
        raise ValueError(f"Metodo segmentazione non supportato: {method}")

    acquisitions = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    global_summary = []

    for acquisition_dir in acquisitions:
        acquisition_id = acquisition_dir.name
        print(f"\nStep 06 - segmentazione pianta: {acquisition_id}")

        out_dir = step06_root / acquisition_id
        out_dir.mkdir(parents=True, exist_ok=True)

        qc_summary_path = step04_root / acquisition_id / "radiometric_calibration_qc_summary.json"

        if not qc_summary_path.exists():
            raise FileNotFoundError(f"Step 04 mancante: {qc_summary_path}")

        with open(qc_summary_path, "r", encoding="utf-8") as f:
            qc_summary = json.load(f)

        if qc_summary["status"] == "FAIL":
            raise RuntimeError(f"{acquisition_id}: Step 04 in FAIL. Non procedo.")

        ndvi_masked_path = step05_root / acquisition_id / "ndvi_masked.npy"
        pseudo_rgb_path = step05_root / acquisition_id / "pseudo_rgb.npy"
        validity_mask_path = step04_root / acquisition_id / "radiometric_validity_mask.npy"

        if not ndvi_masked_path.exists():
            raise FileNotFoundError(f"NDVI masked mancante: {ndvi_masked_path}")

        if not pseudo_rgb_path.exists():
            raise FileNotFoundError(f"Pseudo RGB mancante: {pseudo_rgb_path}")

        if not validity_mask_path.exists():
            raise FileNotFoundError(f"Validity mask mancante: {validity_mask_path}")

        ndvi_masked = np.load(ndvi_masked_path)
        validity_mask = np.load(validity_mask_path)
        pseudo_rgb_img = np.load(pseudo_rgb_path)

        if pseudo_rgb_img.shape[:2] != ndvi_masked.shape:
            raise ValueError(
                f"Dimensione pseudo-RGB {pseudo_rgb_img.shape[:2]} diversa da NDVI {ndvi_masked.shape}"
            )

        valid_ndvi_values = ndvi_masked[np.isfinite(ndvi_masked) & validity_mask]

        threshold_used, otsu_raw = robust_otsu_threshold(
            valid_ndvi_values,
            min_threshold=ndvi_min_threshold,
            max_threshold=ndvi_max_threshold,
        )

        raw_mask = create_initial_mask(
            ndvi_masked=ndvi_masked,
            pseudo_rgb=pseudo_rgb_img,
            validity_mask=validity_mask,
            ndvi_threshold=threshold_used,
            use_exg=use_exg,
            exg_min_threshold=exg_min_threshold,
            use_green_dominance=use_green_dominance,
            green_margin=green_margin,
        )

        clean_mask = clean_plant_mask(
            mask=raw_mask,
            min_object_size_px=min_object_size_px,
            min_hole_size_px=min_hole_size_px,
            opening_radius_px=opening_radius_px,
            closing_radius_px=closing_radius_px,
            erosion_radius_px=erosion_radius_px,
            remove_border_objects=remove_border_objects,
            keep_largest_n_components=keep_largest_n_components,
        )

        np.save(out_dir / "plant_mask_raw.npy", raw_mask)
        np.save(out_dir / "plant_mask_clean.npy", clean_mask)

        save_mask_png(
            raw_mask,
            out_dir / "plant_mask_raw.png",
            f"{acquisition_id} - Plant mask raw"
        )

        save_mask_png(
            clean_mask,
            out_dir / "plant_mask_clean.png",
            f"{acquisition_id} - Plant mask clean"
        )

        save_ndvi_threshold_png(
            ndvi_masked,
            threshold_used,
            out_dir / "ndvi_threshold_contour.png",
            f"{acquisition_id} - NDVI threshold = {threshold_used:.3f}"
        )

        save_exg_png(
            pseudo_rgb_img,
            out_dir / "exg_normalized.png",
            f"{acquisition_id} - Normalized ExG"
        )

        save_overlay_png(
            pseudo_rgb_img,
            raw_mask,
            out_dir / "overlay_raw_mask_on_rgb.png",
            f"{acquisition_id} - Raw mask overlay"
        )

        save_overlay_png(
            pseudo_rgb_img,
            clean_mask,
            out_dir / "overlay_clean_mask_on_rgb.png",
            f"{acquisition_id} - Clean mask overlay"
        )

        n_pixels = clean_mask.size
        n_raw = int(raw_mask.sum())
        n_clean = int(clean_mask.sum())
        n_valid_radiometric = int(validity_mask.sum())

        summary = {
            "created_at": datetime.now().isoformat(),
            "acquisition_id": acquisition_id,
            "method": method,
            "step04_status": qc_summary["status"],

            "thresholding": {
                "otsu_raw_threshold": otsu_raw,
                "threshold_used_after_clipping": threshold_used,
                "ndvi_min_threshold": ndvi_min_threshold,
                "ndvi_max_threshold": ndvi_max_threshold,
            },

            "rgb_filters": {
                "use_exg": use_exg,
                "exg_min_threshold": exg_min_threshold,
                "use_green_dominance": use_green_dominance,
                "green_margin": green_margin,
            },

            "morphology": {
                "min_object_size_px": min_object_size_px,
                "min_hole_size_px": min_hole_size_px,
                "opening_radius_px": opening_radius_px,
                "closing_radius_px": closing_radius_px,
                "erosion_radius_px": erosion_radius_px,
                "remove_border_objects": remove_border_objects,
                "keep_largest_n_components": keep_largest_n_components,
            },

            "pixel_counts": {
                "n_total_pixels": int(n_pixels),
                "n_radiometrically_valid_pixels": n_valid_radiometric,
                "pct_radiometrically_valid_pixels": float(100 * n_valid_radiometric / n_pixels),
                "n_raw_plant_pixels": n_raw,
                "pct_raw_plant_pixels": float(100 * n_raw / n_pixels),
                "n_clean_plant_pixels": n_clean,
                "pct_clean_plant_pixels": float(100 * n_clean / n_pixels),
                "pct_clean_plant_over_radiometric_valid": float(
                    100 * n_clean / n_valid_radiometric
                ) if n_valid_radiometric > 0 else None,
            },

            "outputs": {
                "plant_mask_raw_npy": str(out_dir / "plant_mask_raw.npy"),
                "plant_mask_clean_npy": str(out_dir / "plant_mask_clean.npy"),
                "plant_mask_raw_png": str(out_dir / "plant_mask_raw.png"),
                "plant_mask_clean_png": str(out_dir / "plant_mask_clean.png"),
                "ndvi_threshold_contour_png": str(out_dir / "ndvi_threshold_contour.png"),
                "exg_normalized_png": str(out_dir / "exg_normalized.png"),
                "overlay_raw_mask_on_rgb_png": str(out_dir / "overlay_raw_mask_on_rgb.png"),
                "overlay_clean_mask_on_rgb_png": str(out_dir / "overlay_clean_mask_on_rgb.png"),
            }
        }

        save_json(summary, out_dir / "plant_segmentation_summary.json")
        global_summary.append(summary)

        print(f"Otsu raw threshold: {otsu_raw:.4f}")
        print(f"Threshold usata: {threshold_used:.4f}")
        print(f"Pixel pianta raw: {n_raw} ({100 * n_raw / n_pixels:.2f}%)")
        print(f"Pixel pianta clean: {n_clean} ({100 * n_clean / n_pixels:.2f}%)")
        print(f"Output salvati in: {out_dir}")

    save_json(
        {
            "created_at": datetime.now().isoformat(),
            "n_acquisitions": len(global_summary),
            "summaries": global_summary,
        },
        step06_root / "global_plant_segmentation_summary.json",
    )

    print("\nStep 06 completato.")
    print(f"Output salvati in: {step06_root}")


if __name__ == "__main__":
    main()