import pandas as pd
import os

# =========================
# PATH FILE
# =========================
path = "outputs/07_extracted_plant_pixels/2025-05-30_050/plant_pixel_matrix.csv.gz"

# =========================
# LOAD DATA
# =========================
print("Carico dataset...")
df = pd.read_csv(path)

# =========================
# INFO BASE
# =========================
print("\nShape:", df.shape)

print("\nPrime 10 righe:")
print(df.head(10))

print("\nColonne (prime 10):")
print(df.columns[:10])

# =========================
# VISUALIZZAZIONE PIÙ AMPIA
# =========================
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 1200)

# =========================
# OPZIONE: APRI IN EXCEL
# =========================
open_excel = True

if open_excel:
    preview_path = "preview_for_excel.csv"
    
    # salva solo una parte per evitare crash Excel
    df.head(10000).to_csv(preview_path, index=False)
    
    print(f"\nCreato file preview: {preview_path}")
    
    try:
        os.startfile(preview_path)  # Windows
    except Exception:
        print("Impossibile aprire automaticamente il file")