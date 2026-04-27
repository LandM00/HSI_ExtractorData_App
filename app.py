import streamlit as st
import subprocess
from pathlib import Path
import yaml
from tkinter import Tk, filedialog
import shutil

PYTHON = r"C:/Users/matte/anaconda3/envs/py310/python.exe"
CONFIG_PATH = Path("config.yaml")
TEMP_DATASET_ROOT = Path("_temp_selected_dataset")

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


def select_folder():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory()
    root.destroy()
    return folder


def get_available_acquisitions(dataset_root):
    if dataset_root is None or not dataset_root.exists():
        return []

    return sorted([
        p.name for p in dataset_root.iterdir()
        if p.is_dir()
        and (p / "capture").exists()
        and (p / "results").exists()
    ])


def prepare_dataset_for_processing(original_dataset_root, processing_mode, selected_acquisition):
    original_dataset_root = Path(original_dataset_root)

    if processing_mode == "All acquisitions":
        return original_dataset_root

    if not selected_acquisition:
        raise ValueError("No acquisition selected.")

    if TEMP_DATASET_ROOT.exists():
        shutil.rmtree(TEMP_DATASET_ROOT)

    TEMP_DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    src = original_dataset_root / selected_acquisition
    dst = TEMP_DATASET_ROOT / selected_acquisition

    if not src.exists():
        raise FileNotFoundError(f"Selected acquisition not found: {src}")

    shutil.copytree(src, dst)
    return TEMP_DATASET_ROOT


config = load_config()

if "dataset_root" not in st.session_state:
    st.session_state.dataset_root = config.get("dataset_root", "")

if "output_dir" not in st.session_state:
    st.session_state.output_dir = config.get("output_dir", "outputs")


st.subheader("Dataset settings")

col_ds1, col_ds2 = st.columns([1, 3])

with col_ds1:
    if st.button("Select dataset folder"):
        folder = select_folder()
        if folder:
            st.session_state.dataset_root = folder

with col_ds2:
    st.write("Selected dataset folder:")
    st.code(st.session_state.dataset_root if st.session_state.dataset_root else "No folder selected")

col_out1, col_out2 = st.columns([1, 3])

with col_out1:
    if st.button("Select output folder"):
        folder = select_folder()
        if folder:
            st.session_state.output_dir = folder

with col_out2:
    st.write("Selected output folder:")
    st.code(st.session_state.output_dir)

dataset_root = Path(st.session_state.dataset_root) if st.session_state.dataset_root else None
output_dir = Path(st.session_state.output_dir)

available_acquisitions = get_available_acquisitions(dataset_root)

st.subheader("Processing mode")

processing_mode = st.radio(
    "Choose what to process",
    ["All acquisitions", "Single acquisition"],
    horizontal=True
)

selected_acquisition = None

if processing_mode == "Single acquisition":
    if available_acquisitions:
        selected_acquisition = st.selectbox("Select acquisition", available_acquisitions)
    else:
        st.warning("No valid acquisitions found in the selected dataset folder.")

col_save, col_check = st.columns(2)

with col_save:
    if st.button("Save settings"):
        if dataset_root is None:
            st.error("Select a dataset folder first.")
        else:
            config["dataset_root"] = str(dataset_root)
            config["output_dir"] = str(output_dir)
            config["expected_folders"] = ["capture", "results", "metadata"]
            config["processing_mode_ui"] = {
                "mode": processing_mode,
                "selected_acquisition": selected_acquisition,
                "note": (
                    "The pipeline scripts still process all folders in dataset_root. "
                    "For single acquisition, app.py creates a temporary dataset folder."
                )
            }
            save_config(config)
            st.success("Config updated.")

with col_check:
    if st.button("Check dataset folder"):
        if dataset_root is None or not dataset_root.exists():
            st.error(f"Dataset root not found: {dataset_root}")
        else:
            acquisitions = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
            st.success(f"Dataset found. Acquisitions detected: {len(acquisitions)}")

            for acq_dir in acquisitions:
                capture_ok = (acq_dir / "capture").exists()
                results_ok = (acq_dir / "results").exists()
                metadata_ok = (acq_dir / "metadata").exists()

                status = "OK" if capture_ok and results_ok and metadata_ok else "MISSING"
                st.write(
                    f"- {acq_dir.name}: {status} "
                    f"(capture={capture_ok}, results={results_ok}, metadata={metadata_ok})"
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


def save_current_settings_or_stop():
    if dataset_root is None:
        st.error("Select a dataset folder first.")
        st.stop()

    if not dataset_root.exists():
        st.error(f"Dataset folder does not exist: {dataset_root}")
        st.stop()

    if processing_mode == "Single acquisition" and not selected_acquisition:
        st.error("Select an acquisition first.")
        st.stop()

    try:
        processing_dataset_root = prepare_dataset_for_processing(
            original_dataset_root=dataset_root,
            processing_mode=processing_mode,
            selected_acquisition=selected_acquisition,
        )
    except Exception as e:
        st.error(str(e))
        st.stop()

    config["dataset_root"] = str(processing_dataset_root)
    config["output_dir"] = str(output_dir)
    config["expected_folders"] = ["capture", "results", "metadata"]
    config["processing_mode_ui"] = {
        "mode": processing_mode,
        "original_dataset_root": str(dataset_root),
        "effective_dataset_root": str(processing_dataset_root),
        "selected_acquisition": selected_acquisition,
    }

    save_config(config)
    st.info(f"Effective dataset root: {processing_dataset_root}")


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


st.divider()
st.subheader("Run pipeline")

col1, col2 = st.columns(2)

with col1:
    if st.button("Run full pipeline"):
        save_current_settings_or_stop()

        for name, script in steps.items():
            st.write(f"Running {name}")
            ok = run_script(script)
            if not ok:
                st.stop()

with col2:
    selected_step = st.selectbox("Run single step", list(steps.keys()))

    if st.button("Run selected step"):
        save_current_settings_or_stop()
        run_script(steps[selected_step])


st.divider()
st.subheader("Results preview")

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
        else:
            st.info("Pseudo RGB not available.")

    with col_b:
        if mask_overlay.exists():
            st.image(str(mask_overlay), caption="Plant mask overlay")
        else:
            st.info("Plant mask overlay not available.")

    with col_c:
        if spectral_png.exists():
            st.image(str(spectral_png), caption="Spectral signature")
        else:
            st.info("Spectral signature not available.")

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