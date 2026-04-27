import streamlit as st
import subprocess
from pathlib import Path
import yaml
import shutil
import zipfile
import tempfile
import sys

PYTHON = sys.executable
CONFIG_PATH = Path("config.yaml")
TEMP_SELECTED_DATASET = Path("_temp_selected_dataset")

st.set_page_config(page_title="HSI Plant Pixel Extraction", layout="wide")
st.title("HSI Plant Pixel Extraction Pipeline")


def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def save_config(config):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


config = load_config()


def find_dataset_root(unzipped_root: Path):
    candidates = [unzipped_root] + [p for p in unzipped_root.rglob("*") if p.is_dir()]

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

    if TEMP_SELECTED_DATASET.exists():
        shutil.rmtree(TEMP_SELECTED_DATASET)

    TEMP_SELECTED_DATASET.mkdir(parents=True, exist_ok=True)

    src = dataset_root / selected_acquisition
    dst = TEMP_SELECTED_DATASET / selected_acquisition

    if not src.exists():
        raise FileNotFoundError(f"Selected acquisition not found: {src}")

    shutil.copytree(src, dst)
    return TEMP_SELECTED_DATASET


st.subheader("1. Upload dataset")

uploaded_zip = st.file_uploader(
    "Upload dataset ZIP",
    type=["zip"],
    help="Carica uno ZIP contenente la cartella dataset con le acquisizioni hyperspectral."
)

if "dataset_root" not in st.session_state:
    st.session_state.dataset_root = None

if "output_dir" not in st.session_state:
    st.session_state.output_dir = None

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
            st.error("Dataset non valido: non trovo cartelle con capture/ e results/.")
            st.stop()

        output_dir = work_dir / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        st.session_state.dataset_root = detected_root
        st.session_state.output_dir = output_dir

        st.success("Dataset estratto correttamente.")
        st.write("Dataset root:")
        st.code(str(detected_root))
        st.write("Output directory:")
        st.code(str(output_dir))


dataset_root = st.session_state.dataset_root
output_dir = st.session_state.output_dir

if dataset_root is None:
    st.info("Carica uno ZIP e clicca 'Extract dataset' per iniziare.")
    st.stop()

dataset_root = Path(dataset_root)
output_dir = Path(output_dir)

available_acquisitions = get_available_acquisitions(dataset_root)

st.subheader("2. Dataset check")

if available_acquisitions:
    st.success(f"Acquisizioni trovate: {len(available_acquisitions)}")
    for acq in available_acquisitions:
        st.write(f"- {acq}")
else:
    st.error("Nessuna acquisizione valida trovata.")
    st.stop()


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


steps = {
    "01 - Inspect dataset": "01_inspect_dataset.py",
    "02 - Parse ENVI headers": "02_parse_envi_headers.py",
    "03 - Validate reflectance": "03_load_and_validate_reflectance.py",
    "04 - Radiometric QC + band selection": "04_radiometric_calibration_qc_and_band_selection.py",
    "05 - Create support images": "05_create_support_images.py",
    "06 - Segment plant mask": "06_segment_plant_mask.py",
    "07 - Extract plant pixels": "07_extract_plant_pixels.py",
    "08 - Spectral QC + signature": "08_spectral_qc_and_signature.py",
}


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
    config["processing_mode_ui"] = {
        "mode": processing_mode,
        "selected_acquisition": selected_acquisition,
        "effective_dataset_root": str(effective_dataset_root),
    }

    save_config(config)
    return effective_dataset_root


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
    st.code(result.stdout)
    return True


st.subheader("4. Run pipeline")

col1, col2 = st.columns(2)

with col1:
    if st.button("Run full pipeline"):
        effective_root = save_current_config_or_stop()
        st.info(f"Effective dataset root: {effective_root}")

        for name, script in steps.items():
            st.write(f"Running {name}")
            ok = run_script(script)
            if not ok:
                st.stop()

with col2:
    selected_step = st.selectbox("Run single step", list(steps.keys()))

    if st.button("Run selected step"):
        effective_root = save_current_config_or_stop()
        st.info(f"Effective dataset root: {effective_root}")
        run_script(steps[selected_step])


st.divider()
st.subheader("5. Results preview")

acq_root = output_dir / "07_extracted_plant_pixels"
result_acquisitions = sorted([p.name for p in acq_root.iterdir() if p.is_dir()]) if acq_root.exists() else []

if result_acquisitions:
    acq = st.selectbox("Acquisition results", result_acquisitions)

    col_a, col_b, col_c = st.columns(3)

    pseudo_rgb = output_dir / "05_support_images" / acq / "pseudo_rgb.png"
    mask_overlay = output_dir / "06_plant_segmentation" / acq / "overlay_clean_mask_on_rgb.png"
    spectral_png = output_dir / "08_spectral_qc" / acq / "spectral_signature_mean_p05_p95.png"

    with col_a:
        if pseudo_rgb.exists():
            st.image(str(pseudo_rgb), caption="Pseudo RGB")

    with col_b:
        if mask_overlay.exists():
            st.image(str(mask_overlay), caption="Plant mask overlay")

    with col_c:
        if spectral_png.exists():
            st.image(str(spectral_png), caption="Spectral signature")

    st.subheader("Download")

    csv_gz = output_dir / "07_extracted_plant_pixels" / acq / "plant_pixel_matrix.csv.gz"
    npz = output_dir / "07_extracted_plant_pixels" / acq / "plant_pixel_spectra.npz"
    mean_csv = output_dir / "07_extracted_plant_pixels" / acq / "plant_mean_spectrum.csv"
    summary_json = output_dir / "07_extracted_plant_pixels" / acq / "extraction_summary.json"

    if csv_gz.exists():
        st.download_button(
            "Download pixel matrix CSV.GZ",
            data=csv_gz.read_bytes(),
            file_name=f"{acq}_plant_pixel_matrix.csv.gz",
            mime="application/gzip"
        )

    if npz.exists():
        st.download_button(
            "Download NPZ scientific file",
            data=npz.read_bytes(),
            file_name=f"{acq}_plant_pixel_spectra.npz",
            mime="application/octet-stream"
        )

    if mean_csv.exists():
        st.download_button(
            "Download mean spectrum CSV",
            data=mean_csv.read_bytes(),
            file_name=f"{acq}_plant_mean_spectrum.csv",
            mime="text/csv"
        )

    if spectral_png.exists():
        st.download_button(
            "Download spectral signature PNG",
            data=spectral_png.read_bytes(),
            file_name=f"{acq}_spectral_signature.png",
            mime="image/png"
        )

    if summary_json.exists():
        st.download_button(
            "Download extraction summary JSON",
            data=summary_json.read_bytes(),
            file_name=f"{acq}_extraction_summary.json",
            mime="application/json"
        )

else:
    st.info("No extraction output found. Run the pipeline first.")