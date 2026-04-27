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
    """
    Calcola soglia Otsu sui valori validi NDVI.
    Poi vincola la soglia entro un range agronomicamente plausibile.
    """
    values = np.asarray(values)
    values = values[np.isfinite(values)]

    if values.size == 0:
        raise RuntimeError("Nessun valore NDVI valido per calcolare Otsu.")

    otsu = float(threshold_otsu(values))

    threshold = float(np.clip(otsu, min_threshold, max_threshold))

    return threshold, otsu


def create_initial_plant_mask(ndvi_masked, validity_mask, threshold):
    """
    Maschera iniziale:
    pianta = NDVI >= threshold AND pixel radiometricamente valido
    """
    mask = (
        np.isfinite(ndvi_masked)
        & validity_mask
        & (ndvi_masked >= threshold)
    )

    return mask.astype(bool)


def clean_plant_mask(
    mask,
    min_object_size_px=300,
    min_hole_size_px=300,
    erosion_radius_px=1,
    remove_border_objects=False,
):
    """
    Pulizia morfologica robusta della maschera pianta.
    """
    clean = mask.astype(bool)

    if remove_border_objects:
        clean = clear_border(clean)

    clean = remove_small_objects(clean, min_size=min_object_size_px)

    clean = binary_closing(clean, footprint=disk(2))
    clean = binary_opening(clean, footprint=disk(1))

    clean = remove_small_holes(clean, area_threshold=min_hole_size_px)

    if erosion_radius_px is not None and erosion_radius_px > 0:
        clean = binary_erosion(clean, footprint=disk(erosion_radius_px))

    clean = remove_small_objects(clean, min_size=min_object_size_px)

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
    """
    Overlay semplice: maschera in rosso sopra RGB.
    """
    bg = background_rgb.copy()
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

    method = seg_cfg.get("method", "ndvi_otsu")
    ndvi_min_threshold = float(seg_cfg.get("ndvi_min_threshold", 0.20))
    ndvi_max_threshold = float(seg_cfg.get("ndvi_max_threshold", 0.75))
    min_object_size_px = int(seg_cfg.get("min_object_size_px", 300))
    min_hole_size_px = int(seg_cfg.get("min_hole_size_px", 300))
    erosion_radius_px = int(seg_cfg.get("erosion_radius_px", 1))
    remove_border_objects = bool(seg_cfg.get("remove_border_objects", False))

    if method != "ndvi_otsu":
        raise ValueError(f"Metodo segmentazione non supportato: {method}")

    acquisitions = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    global_summary = []

    for acquisition_dir in acquisitions:
        acquisition_id = acquisition_dir.name
        print(f"\nStep 06 - segmentazione pianta: {acquisition_id}")

        out_dir = step06_root / acquisition_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------
        # Load previous outputs
        # -------------------------
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

        ndvi_masked = np.load(ndvi_masked_path)
        validity_mask = np.load(validity_mask_path)

        # Per overlay carichiamo l'array RGB originale 512x512 generato nello step 05
        pseudo_rgb_img = np.load(pseudo_rgb_path)

        if pseudo_rgb_img.shape[:2] != ndvi_masked.shape:
            raise ValueError(
                f"Dimensione pseudo-RGB {pseudo_rgb_img.shape[:2]} diversa da NDVI {ndvi_masked.shape}"
            )

        # -------------------------
        # Threshold
        # -------------------------
        valid_ndvi_values = ndvi_masked[np.isfinite(ndvi_masked) & validity_mask]

        threshold_used, otsu_raw = robust_otsu_threshold(
            valid_ndvi_values,
            min_threshold=ndvi_min_threshold,
            max_threshold=ndvi_max_threshold,
        )

        raw_mask = create_initial_plant_mask(
            ndvi_masked=ndvi_masked,
            validity_mask=validity_mask,
            threshold=threshold_used,
        )

        clean_mask = clean_plant_mask(
            mask=raw_mask,
            min_object_size_px=min_object_size_px,
            min_hole_size_px=min_hole_size_px,
            erosion_radius_px=erosion_radius_px,
            remove_border_objects=remove_border_objects,
        )

        # -------------------------
        # Save arrays
        # -------------------------
        np.save(out_dir / "plant_mask_raw.npy", raw_mask)
        np.save(out_dir / "plant_mask_clean.npy", clean_mask)

        # -------------------------
        # Save visual outputs
        # -------------------------
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

        # -------------------------
        # Summary
        # -------------------------
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

            "morphology": {
                "min_object_size_px": min_object_size_px,
                "min_hole_size_px": min_hole_size_px,
                "erosion_radius_px": erosion_radius_px,
                "remove_border_objects": remove_border_objects,
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