import streamlit as st
import subprocess
from pathlib import Path
import yaml
import zipfile
import tempfile
import shutil
import sys
import os
import io

PYTHON = sys.executable
CONFIG_PATH = Path("config.yaml")

st.set_page_config(page_title="HSI Plant Pixel Extraction", layout="wide")
st.title("HSI Plant Pixel Extraction Pipeline")


# ============================================================
# CONFIG
# ============================================================

def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def save_config(config):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


config = load_config()


# ============================================================
# HELPERS
# ============================================================

def find_dataset_root(root: Path):
    candidates = [root] + [p for p in root.rglob("*") if p.is_dir()]

    for candidate in candidates:
        acquisitions = [
            p for p in candidate.iterdir()
            if p.is_dir()
            and (p / "capture").exists()
            and (p / "results").exists()
        ]
        if acquisitions:
            return candidate

    return None


def get_available_acquisitions(dataset_root):
    if dataset_root is None or not dataset_root.exists():
        return []

    return sorted([
        p.name for p in dataset_root.iterdir()
        if p.is_dir()
        and (p / "capture").exists()
        and (p / "results").exists()
    ])


def prepare_dataset_for_processing(dataset_root, mode, selected_acquisition):
    dataset_root = Path(dataset_root)

    if mode == "All acquisitions":
        return dataset_root

    if not selected_acquisition:
        raise ValueError("No acquisition selected.")

    work_dir = Path(st.session_state.work_dir)
    temp_selected_dataset = work_dir / "_temp_selected_dataset"

    if temp_selected_dataset.exists():
        shutil.rmtree(temp_selected_dataset)

    temp_selected_dataset.mkdir(parents=True, exist_ok=True)

    src = dataset_root / selected_acquisition
    dst = temp_selected_dataset / selected_acquisition

    if not src.exists():
        raise FileNotFoundError(f"Selected acquisition not found: {src}")

    shutil.copytree(src, dst)
    return temp_selected_dataset


def run_script(script):
    result = subprocess.run(
        [PYTHON, script],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        st.error(f"Error in {script}")
        st.code(result.stderr)
        return False

    st.success(f"{script} completed")

    if result.stdout.strip():
        with st.expander(f"Log {script}"):
            st.code(result.stdout)

    return True


def save_current_config_or_stop():
    try:
        effective_dataset_root = prepare_dataset_for_processing(
            dataset_root=dataset_root,
            mode=processing_mode,
            selected_acquisition=selected_acquisition,
        )
    except Exception as e:
        st.error(str(e))
        st.stop()

    config["dataset_root"] = str(effective_dataset_root)
    config["output_dir"] = str(output_dir)
    config["expected_folders"] = ["capture", "results", "metadata"]

    config.setdefault("radiometric_qc", {})
    config["radiometric_qc"].setdefault("min_reflectance", 0.0)
    config["radiometric_qc"].setdefault("max_reflectance", 1.5)
    config["radiometric_qc"].setdefault("min_pct_valid_reflectance", 95.0)
    config["radiometric_qc"].setdefault("min_white_minus_dark_median_dn", 20.0)
    config["radiometric_qc"].setdefault("fail_min_valid_bands", 40)
    config["radiometric_qc"].setdefault("warning_min_valid_bands", 80)
    config["radiometric_qc"].setdefault("fail_min_valid_pixels_pct", 50.0)
    config["radiometric_qc"].setdefault("warning_min_valid_pixels_pct", 80.0)
    config["radiometric_qc"].setdefault("max_allowed_median_abs_diff", 0.00001)

    config["cleaning"] = {
        "method": "value_range_to_nan",
        "min_reflectance": float(clean_min),
        "max_reflectance": float(clean_max),
    }

    config["processing_mode_ui"] = {
        "mode": processing_mode,
        "selected_acquisition": selected_acquisition,
        "effective_dataset_root": str(effective_dataset_root),
    }

    save_config(config)
    return effective_dataset_root


def copy_if_exists(src, dst_dir, new_name=None):
    src = Path(src)
    dst_dir = Path(dst_dir)

    if src.exists() and src.is_file():
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_name = new_name if new_name else src.name
        shutil.copy2(src, dst_dir / dst_name)
        return 1

    return 0


def organize_selected_outputs(output_root: Path, export_options: dict):
    output_root = Path(output_root)

    step07_root = output_root / "07_extracted_plant_pixels"
    step08_root = output_root / "08_spectral_qc"
    organized_root = output_root / "HSI_outputs"

    if organized_root.exists():
        shutil.rmtree(organized_root)

    organized_root.mkdir(parents=True, exist_ok=True)

    if not step07_root.exists():
        return organized_root, 0

    acquisitions = sorted([p for p in step07_root.iterdir() if p.is_dir()])
    copied_count = 0

    for acq_dir in acquisitions:
        acq = acq_dir.name

        raw_dir = organized_root / acq / "RAW_FULL"
        raw_qc_dir = step08_root / acq / "RAW_FULL"

        if export_options.get("raw_matrix_csv_gz", False):
            copied_count += copy_if_exists(
                acq_dir / "plant_pixel_matrix_RAW_FULL.csv.gz",
                raw_dir
            )

        if export_options.get("raw_npz", False):
            copied_count += copy_if_exists(
                acq_dir / "plant_pixel_spectra_RAW_FULL.npz",
                raw_dir
            )

        if export_options.get("raw_statistics_csv", False):
            copied_count += copy_if_exists(
                raw_qc_dir / "extracted_spectra_statistics.csv",
                raw_dir,
                new_name="RAW_FULL_spectral_statistics.csv"
            )

        if export_options.get("raw_signature_png", False):
            copied_count += copy_if_exists(
                raw_qc_dir / "spectral_signature_mean_p05_p95.png",
                raw_dir,
                new_name="RAW_FULL_spectral_signature_mean_p05_p95.png"
            )

        clean_dir = organized_root / acq / "CLEAN_NAN"
        clean_qc_dir = step08_root / acq / "CLEAN_NAN"

        if export_options.get("clean_matrix_csv_gz", False):
            copied_count += copy_if_exists(
                acq_dir / "plant_pixel_matrix_CLEAN_NAN.csv.gz",
                clean_dir
            )

        if export_options.get("clean_npz", False):
            copied_count += copy_if_exists(
                acq_dir / "plant_pixel_spectra_CLEAN_NAN.npz",
                clean_dir
            )

        if export_options.get("clean_statistics_csv", False):
            copied_count += copy_if_exists(
                clean_qc_dir / "extracted_spectra_statistics.csv",
                clean_dir,
                new_name="CLEAN_NAN_spectral_statistics.csv"
            )

        if export_options.get("clean_signature_png", False):
            copied_count += copy_if_exists(
                clean_qc_dir / "spectral_signature_mean_p05_p95.png",
                clean_dir,
                new_name="CLEAN_NAN_spectral_signature_mean_p05_p95.png"
            )

    return organized_root, copied_count


def build_zip_from_folder(folder_path: Path):
    folder_path = Path(folder_path)
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for file_path in folder_path.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(folder_path.parent)
                z.write(file_path, arcname=str(arcname))

    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# SESSION STATE
# ============================================================

if "dataset_root" not in st.session_state:
    st.session_state.dataset_root = None

if "output_dir" not in st.session_state:
    st.session_state.output_dir = None

if "pipeline_ran" not in st.session_state:
    st.session_state.pipeline_ran = False

if "work_dir" not in st.session_state:
    st.session_state.work_dir = None

if "organized_output_dir" not in st.session_state:
    st.session_state.organized_output_dir = None


# ============================================================
# 1. DATASET UPLOAD
# ============================================================

st.subheader("1. Upload dataset ZIP")

uploaded_zip = st.file_uploader(
    "Upload dataset ZIP",
    type=["zip"],
    help="Carica uno ZIP contenente acquisizioni con capture/, results/ e metadata/."
)

if uploaded_zip is not None:
    if st.button("Extract dataset"):
        work_dir = Path(tempfile.mkdtemp())
        zip_path = work_dir / "dataset.zip"

        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.getbuffer())

        extract_dir = work_dir / "dataset"
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

        detected_root = find_dataset_root(extract_dir)

        if detected_root is None:
            st.error("Dataset non valido: non trovo acquisizioni con capture/ e results/.")
            st.stop()

        output_dir_temp = work_dir / "outputs"
        output_dir_temp.mkdir(parents=True, exist_ok=True)

        st.session_state.work_dir = work_dir
        st.session_state.dataset_root = detected_root
        st.session_state.output_dir = output_dir_temp
        st.session_state.pipeline_ran = False
        st.session_state.organized_output_dir = None

        st.success("Dataset estratto correttamente.")
        st.write("Dataset root:")
        st.code(str(detected_root))
        st.write("Output directory:")
        st.code(str(output_dir_temp))


dataset_root = st.session_state.dataset_root
output_dir = st.session_state.output_dir

if dataset_root is None:
    st.info("Carica uno ZIP e clicca 'Extract dataset' per iniziare.")
    st.stop()

dataset_root = Path(dataset_root)
output_dir = Path(output_dir)


# ============================================================
# 2. DATASET CHECK
# ============================================================

st.subheader("2. Dataset check")

available_acquisitions = get_available_acquisitions(dataset_root)

if available_acquisitions:
    st.success(f"Acquisizioni trovate: {len(available_acquisitions)}")
    for acq_name in available_acquisitions:
        st.write(f"- {acq_name}")
else:
    st.error("Nessuna acquisizione valida trovata.")
    st.stop()


# ============================================================
# 3. PROCESSING MODE
# ============================================================

st.subheader("3. Processing mode")

processing_mode = st.radio(
    "Choose what to process",
    ["All acquisitions", "Single acquisition"],
    horizontal=True
)

selected_acquisition = None

if processing_mode == "Single acquisition":
    selected_acquisition = st.selectbox(
        "Select acquisition",
        available_acquisitions
    )


# ============================================================
# 4. CLEANING SETTINGS
# ============================================================

st.subheader("4. Cleaning settings")

clean_cfg = config.get("cleaning", {})

col_c1, col_c2 = st.columns(2)

with col_c1:
    clean_min = st.number_input(
        "Min reflectance",
        value=float(clean_cfg.get("min_reflectance", 0.0)),
        step=0.1
    )

with col_c2:
    clean_max = st.number_input(
        "Max reflectance",
        value=float(clean_cfg.get("max_reflectance", 1.5)),
        step=0.1
    )


# ============================================================
# 5. EXPORT OPTIONS
# ============================================================

st.subheader("5. Export options")

st.caption(
    "Seleziona quali file vuoi salvare nella cartella finale ordinata HSI_outputs."
)

col_raw_opts, col_clean_opts = st.columns(2)

with col_raw_opts:
    st.markdown("### RAW_FULL")
    raw_matrix_csv_gz = st.checkbox("RAW_FULL pixel matrix CSV.GZ", value=True)
    raw_npz = st.checkbox("RAW_FULL scientific NPZ", value=True)
    raw_statistics_csv = st.checkbox("RAW_FULL statistics CSV", value=True)
    raw_signature_png = st.checkbox("RAW_FULL spectral signature PNG", value=True)

with col_clean_opts:
    st.markdown("### CLEAN_NAN")
    clean_matrix_csv_gz = st.checkbox("CLEAN_NAN pixel matrix CSV.GZ", value=True)
    clean_npz = st.checkbox("CLEAN_NAN scientific NPZ", value=True)
    clean_statistics_csv = st.checkbox("CLEAN_NAN statistics CSV", value=True)
    clean_signature_png = st.checkbox("CLEAN_NAN spectral signature PNG", value=True)

export_options = {
    "raw_matrix_csv_gz": raw_matrix_csv_gz,
    "raw_npz": raw_npz,
    "raw_statistics_csv": raw_statistics_csv,
    "raw_signature_png": raw_signature_png,
    "clean_matrix_csv_gz": clean_matrix_csv_gz,
    "clean_npz": clean_npz,
    "clean_statistics_csv": clean_statistics_csv,
    "clean_signature_png": clean_signature_png,
}


# ============================================================
# 6. RUN ANALYSIS
# ============================================================

steps = [
    ("01 - Inspect dataset", "01_inspect_dataset.py"),
    ("02 - Parse ENVI headers", "02_parse_envi_headers.py"),
    ("03 - Validate reflectance", "03_load_and_validate_reflectance.py"),
    ("04 - Radiometric QC", "04_radiometric_calibration_qc_and_band_selection.py"),
    ("05 - Create support images", "05_create_support_images.py"),
    ("06 - Segment plant mask", "06_segment_plant_mask.py"),
    ("07a - Extract RAW_FULL plant pixels", "07a_extract_raw_full.py"),
    ("07b - Create CLEAN_NAN dataset", "07b_clean_raw_full_to_nan.py"),
    ("08 - Spectral QC + signature", "08_spectral_qc_and_signature.py"),
]

st.subheader("6. Run analysis")

if st.button("Run analysis", type="primary"):
    effective_root = save_current_config_or_stop()

    st.info(f"Effective dataset root: {effective_root}")
    st.info(f"Output directory: {output_dir}")

    progress = st.progress(0)
    status_box = st.empty()

    for i, (name, script) in enumerate(steps, start=1):
        status_box.write(f"Running {name}")
        ok = run_script(script)

        progress.progress(i / len(steps))

        if not ok:
            st.session_state.pipeline_ran = False
            st.stop()

    organized_root, copied_count = organize_selected_outputs(
        output_root=output_dir,
        export_options=export_options
    )

    st.session_state.organized_output_dir = organized_root
    st.session_state.pipeline_ran = True

    st.success("Analysis completed successfully.")
    st.success(f"Organized outputs created. Files copied: {copied_count}")
    st.code(str(organized_root))


# ============================================================
# 7. RESULTS PREVIEW
# ============================================================

st.divider()
st.subheader("7. Results preview")

if not st.session_state.pipeline_ran:
    st.info("Run analysis to generate and view results.")
    st.stop()

acq_root = output_dir / "07_extracted_plant_pixels"
result_acquisitions = sorted([p.name for p in acq_root.iterdir() if p.is_dir()]) if acq_root.exists() else []

if not result_acquisitions:
    st.info("No extraction output found. Run the analysis first.")
    st.stop()

acq = st.selectbox("Acquisition results", result_acquisitions)

dataset_version = st.radio(
    "Preview dataset version",
    ["CLEAN_NAN", "RAW_FULL"],
    horizontal=True
)

col_a, col_b, col_c = st.columns(3)

pseudo_rgb = output_dir / "05_support_images" / acq / "pseudo_rgb.png"
mask_overlay = output_dir / "06_plant_segmentation" / acq / "overlay_clean_mask_on_rgb.png"
spectral_png = output_dir / "08_spectral_qc" / acq / dataset_version / "spectral_signature_mean_p05_p95.png"

with col_a:
    if pseudo_rgb.exists():
        st.image(str(pseudo_rgb), caption="Pseudo RGB")
    else:
        st.info("Pseudo RGB not available.")

with col_b:
    if mask_overlay.exists():
        st.image(str(mask_overlay), caption="Plant mask overlay")
    else:
        st.info("Plant mask overlay not available.")

with col_c:
    if spectral_png.exists():
        st.image(str(spectral_png), caption=f"{dataset_version} spectral signature")
    else:
        st.info(f"{dataset_version} spectral signature not available.")


# ============================================================
# 8. FINAL OUTPUT FOLDER + DOWNLOAD
# ============================================================

st.subheader("8. Final organized output folder")

organized_output_dir = st.session_state.organized_output_dir

if organized_output_dir:
    organized_output_dir = Path(organized_output_dir)

    st.write("Final folder:")
    st.code(str(organized_output_dir))

    st.info(
        "I file selezionati sono stati copiati nella cartella HSI_outputs, "
        "organizzati per acquisizione e versione dati."
    )

    if organized_output_dir.exists():
        zip_bytes = build_zip_from_folder(organized_output_dir)

        st.download_button(
            label="Download HSI_outputs ZIP",
            data=zip_bytes,
            file_name="HSI_outputs.zip",
            mime="application/zip"
        )

    if sys.platform.startswith("win"):
        if st.button("Open HSI_outputs folder"):
            os.startfile(organized_output_dir)

else:
    st.info("No organized output folder available yet.")