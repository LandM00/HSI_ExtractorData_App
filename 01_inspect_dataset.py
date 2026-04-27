from pathlib import Path
import csv
import yaml
import json
from datetime import datetime


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_files(acquisition_dir):
    capture_dir = acquisition_dir / "capture"
    results_dir = acquisition_dir / "results"
    metadata_dir = acquisition_dir / "metadata"

    return {
        "acquisition_id": acquisition_dir.name,
        "acquisition_path": str(acquisition_dir),

        "capture_dir_exists": capture_dir.exists(),
        "results_dir_exists": results_dir.exists(),
        "metadata_dir_exists": metadata_dir.exists(),

        "raw_file": next(capture_dir.glob("*.raw"), None) if capture_dir.exists() else None,
        "raw_hdr": next(capture_dir.glob("*.hdr"), None) if capture_dir.exists() else None,

        "dark_raw": next(capture_dir.glob("DARKREF_*.raw"), None) if capture_dir.exists() else None,
        "dark_hdr": next(capture_dir.glob("DARKREF_*.hdr"), None) if capture_dir.exists() else None,

        "white_raw": next(capture_dir.glob("WHITEREF_*.raw"), None) if capture_dir.exists() else None,
        "white_hdr": next(capture_dir.glob("WHITEREF_*.hdr"), None) if capture_dir.exists() else None,

        "reflectance_dat": next(results_dir.glob("REFLECTANCE_*.dat"), None) if results_dir.exists() else None,
        "reflectance_hdr": next(results_dir.glob("REFLECTANCE_*.hdr"), None) if results_dir.exists() else None,
    }


def file_size_mb(path):
    if path is None or not Path(path).exists():
        return None
    return round(Path(path).stat().st_size / (1024 * 1024), 3)


def serialize_record(record):
    out = {}

    for key, value in record.items():
        if isinstance(value, Path):
            out[key] = str(value)
            out[key + "_size_mb"] = file_size_mb(value)
        else:
            out[key] = value

    required = [
        "raw_file",
        "raw_hdr",
        "dark_raw",
        "dark_hdr",
        "white_raw",
        "white_hdr",
        "reflectance_dat",
        "reflectance_hdr",
    ]

    missing = [k for k in required if record.get(k) is None]
    out["missing_files"] = "; ".join(missing)
    out["is_complete"] = len(missing) == 0

    return out


def inspect_dataset(dataset_root):
    dataset_root = Path(dataset_root)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root non trovata: {dataset_root}")

    acquisitions = sorted(
        [p for p in dataset_root.iterdir() if p.is_dir()]
    )

    records = []

    for acquisition_dir in acquisitions:
        record = find_files(acquisition_dir)
        records.append(serialize_record(record))

    return records


def save_csv(records, output_path):
    if not records:
        raise ValueError("Nessuna acquisizione trovata.")

    keys = sorted(records[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def save_json(records, output_path):
    payload = {
        "created_at": datetime.now().isoformat(),
        "n_acquisitions": len(records),
        "records": records,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    config = load_config()

    dataset_root = Path(config["dataset_root"])
    output_dir = Path(config["output_dir"]) / "01_dataset_inspection"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = inspect_dataset(dataset_root)

    csv_path = output_dir / "dataset_inventory.csv"
    json_path = output_dir / "dataset_inventory.json"

    save_csv(records, csv_path)
    save_json(records, json_path)

    print("\nDataset inspection completata.")
    print(f"Acquisizioni trovate: {len(records)}")
    print(f"CSV salvato in: {csv_path}")
    print(f"JSON salvato in: {json_path}")

    print("\nSintesi:")
    for r in records:
        status = "OK" if r["is_complete"] else "MANCANO FILE"
        print(f"- {r['acquisition_id']}: {status}")
        if not r["is_complete"]:
            print(f"  Mancanti: {r['missing_files']}")


if __name__ == "__main__":
    main()