from pathlib import Path
import json
import csv
import yaml
from datetime import datetime


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_envi_header(hdr_path):
    """
    Parser semplice e robusto per file header ENVI.
    Gestisce anche campi multilinea tra graffe, es. wavelength = { ... }.
    """
    hdr_path = Path(hdr_path)

    if not hdr_path.exists():
        raise FileNotFoundError(f"Header non trovato: {hdr_path}")

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
    """
    Converte valori tipo:
    { 397.32, 400.21, ... }
    in lista di stringhe.
    """
    if value is None:
        return []

    value = value.replace("{", "").replace("}", "")
    parts = [p.strip() for p in value.replace("\n", " ").split(",")]
    return [p for p in parts if p != ""]


def to_int(value):
    try:
        return int(str(value).strip())
    except Exception:
        return None


def to_float_list(value):
    out = []
    for item in clean_brace_list(value):
        try:
            out.append(float(item))
        except Exception:
            pass
    return out


def summarize_header(hdr_path, acquisition_id, file_role):
    metadata = parse_envi_header(hdr_path)

    wavelengths = to_float_list(metadata.get("wavelength"))

    summary = {
        "acquisition_id": acquisition_id,
        "file_role": file_role,
        "hdr_path": str(hdr_path),

        "samples": to_int(metadata.get("samples")),
        "lines": to_int(metadata.get("lines")),
        "bands": to_int(metadata.get("bands")),
        "header_offset": to_int(metadata.get("header offset")),
        "file_type": metadata.get("file type"),
        "data_type": metadata.get("data type"),
        "interleave": metadata.get("interleave"),
        "byte_order": metadata.get("byte order"),

        "wavelength_units": metadata.get("wavelength units"),
        "n_wavelengths": len(wavelengths),
        "wavelength_min_nm": min(wavelengths) if wavelengths else None,
        "wavelength_max_nm": max(wavelengths) if wavelengths else None,
        "wavelength_first_nm": wavelengths[0] if wavelengths else None,
        "wavelength_last_nm": wavelengths[-1] if wavelengths else None,

        "band_names_present": "band names" in metadata,
        "raw_metadata": metadata,
    }

    if summary["bands"] is not None and wavelengths:
        summary["bands_match_wavelengths"] = summary["bands"] == len(wavelengths)
    else:
        summary["bands_match_wavelengths"] = None

    return summary


def find_acquisition_headers(acquisition_dir):
    acquisition_dir = Path(acquisition_dir)

    capture_dir = acquisition_dir / "capture"
    results_dir = acquisition_dir / "results"

    headers = []

    raw_hdr = capture_dir / f"{acquisition_dir.name}.hdr"
    if raw_hdr.exists():
        headers.append(("raw", raw_hdr))

    dark_hdrs = sorted(capture_dir.glob("DARKREF_*.hdr"))
    for hdr in dark_hdrs:
        headers.append(("dark_reference", hdr))

    white_hdrs = sorted(capture_dir.glob("WHITEREF_*.hdr"))
    for hdr in white_hdrs:
        headers.append(("white_reference", hdr))

    reflectance_hdrs = sorted(results_dir.glob("REFLECTANCE_*.hdr"))
    for hdr in reflectance_hdrs:
        headers.append(("reflectance", hdr))

    return headers


def save_csv(records, output_path):
    flat_records = []

    for r in records:
        flat = {k: v for k, v in r.items() if k != "raw_metadata"}
        flat_records.append(flat)

    keys = sorted(flat_records[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flat_records)


def save_json(records, output_path):
    payload = {
        "created_at": datetime.now().isoformat(),
        "n_headers": len(records),
        "records": records,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    config = load_config()

    dataset_root = Path(config["dataset_root"])
    output_dir = Path(config["output_dir"]) / "02_header_parsing"
    output_dir.mkdir(parents=True, exist_ok=True)

    acquisitions = sorted([p for p in dataset_root.iterdir() if p.is_dir()])

    records = []

    for acquisition_dir in acquisitions:
        headers = find_acquisition_headers(acquisition_dir)

        for file_role, hdr_path in headers:
            summary = summarize_header(
                hdr_path=hdr_path,
                acquisition_id=acquisition_dir.name,
                file_role=file_role,
            )
            records.append(summary)

    if not records:
        raise RuntimeError("Nessun file .hdr trovato.")

    csv_path = output_dir / "envi_header_summary.csv"
    json_path = output_dir / "envi_header_full_metadata.json"

    save_csv(records, csv_path)
    save_json(records, json_path)

    print("\nParsing header ENVI completato.")
    print(f"Header letti: {len(records)}")
    print(f"CSV salvato in: {csv_path}")
    print(f"JSON completo salvato in: {json_path}")

    print("\nSintesi:")
    for r in records:
        print(
            f"- {r['acquisition_id']} | {r['file_role']} | "
            f"{r['lines']} x {r['samples']} x {r['bands']} | "
            f"interleave={r['interleave']} | "
            f"data_type={r['data_type']} | "
            f"wavelengths={r['n_wavelengths']} | "
            f"match={r['bands_match_wavelengths']}"
        )


if __name__ == "__main__":
    main()