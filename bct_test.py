# -------------------------------
# Automatische Paketprüfung + Installation
# -------------------------------
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_packages = ["pandas", "numpy", "bctpy", "openpyxl", "scipy", "statsmodels"]

for pkg in required_packages:
    try:
        __import__(pkg)
    except ImportError:
        print(f"{pkg} fehlt, wird installiert...")
        install(pkg)

# -------------------------------
# Sichere Imports
# -------------------------------
import os
import glob
import numpy as np
import pandas as pd
import bct
from scipy.stats import f_oneway
from statsmodels.stats.anova import AnovaRM
# Pfad zu Testmatrizen
# -------------------------------
root = r"C:\Users\timo-\Desktop\Forschung\bctpy_mrtrix\Test_matrizen"
sessions = ["ses-1", "ses-2", "ses-3", "ses-4"]

results_dir = os.path.join(r"C:\Users\timo-\Desktop\Forschung\bctpy_mrtrix", "results")
os.makedirs(results_dir, exist_ok=True)
# -------------------------------
# XLSX → NPY Konvertierung (falls nötig)
# -------------------------------
for ses in sessions:
    ses_folder = os.path.join(root, ses)
    if not os.path.isdir(ses_folder):
        continue

    xlsx_files = glob.glob(os.path.join(ses_folder, "*.xlsx"))
    npy_files = glob.glob(os.path.join(ses_folder, "*.npy"))

    # Nur konvertieren, wenn es XLSX gibt aber noch keine NPY
    if xlsx_files and not npy_files:
        print(f"🔄 Konvertiere XLSX → NPY in {ses_folder}")

        for file in xlsx_files:
            name = os.path.splitext(os.path.basename(file))[0]

            df_mat = pd.read_excel(file, header=None)
            df_mat = df_mat.dropna(axis=0, how="all").dropna(axis=1, how="all")
            A = df_mat.to_numpy(dtype=float)

            # Sicherheitscheck: quadratische Matrix
            if A.shape[0] != A.shape[1]:
                raise ValueError(f"❌ Matrix nicht quadratisch: {file}")

            npy_path = os.path.join(ses_folder, f"{name}.npy")
            np.save(npy_path, A)

            print(f"✅ Gespeichert: {npy_path}")

# -------------------------------

all_data = []

for ses in sessions:
    ses_folder = os.path.join(root, ses)
    files = glob.glob(os.path.join(ses_folder, "*.npy"))
    if len(files) == 0:
        print(f"⚠️ Warnung: Keine Dateien gefunden in {ses_folder}")
        continue

    for f in files:
        subject = os.path.splitext(os.path.basename(f))[0].split("_")[0]
        A = np.load(f)
        A_bin = (A > 0).astype(int)

        # Globale Werte
        avg_degree = np.mean(bct.degrees_und(A_bin))
        avg_strength = np.mean(bct.strengths_und(A))

        # Nodalwerte
        degree_vec = bct.degrees_und(A_bin)
        strength_vec = bct.strengths_und(A)

        data_row = {
            "subject": subject,
            "session": ses,
            "avg_degree": avg_degree,
            "avg_strength": avg_strength
        }

        for i, val in enumerate(degree_vec):
            data_row[f"deg_region_{i + 1}"] = val
        for i, val in enumerate(strength_vec):
            data_row[f"str_region_{i + 1}"] = val

        all_data.append(data_row)

# -------------------------------
# DataFrame erstellen
# -------------------------------
df = pd.DataFrame(all_data)

if df.empty:
    raise RuntimeError("❌ Keine Daten gefunden! Prüfe den Pfad zu Testmatrizen und Session-Ordnern.")

# Ergebnisse speichern
xlsx_file = os.path.join(results_dir, "graph_metrics_all_sessions.xlsx")
df.to_excel(xlsx_file, index=False)
print(f"✅ Datei gespeichert: {xlsx_file}")
print("Spalten in df:", df.columns.tolist())
print(df.head())

# -------------------------------
# ANOVA global
# -------------------------------
deg_ses = [df[df["session"]==s]["avg_degree"] for s in sessions if s in df["session"].values]
f_val, p_val = f_oneway(*deg_ses)
print(f"Global Degree ANOVA: F={f_val:.3f}, p={p_val:.3f}")

# -------------------------------
# Nodal ANOVAs
# -------------------------------
anova_txt = os.path.join(results_dir, "anova_results.txt")
excel_rows = []

with open(anova_txt, "w") as f:
    f.write(f"Global Degree ANOVA: F={f_val:.3f}, p={p_val:.3f}\n\n")
    excel_rows.append({"type":"global","region":"all","F_value":f_val,"p_value":p_val})

    for i in range(len(degree_vec)):
        col = f"deg_region_{i+1}"
        aov = AnovaRM(df, depvar=col, subject="subject", within=["session"]).fit()
        header = f"Region {i+1} ANOVA:\n"
        f.write(header)
        f.write(aov.summary().as_text() + "\n\n")
        print(header)
        print(aov.summary())

        aov_table = aov.anova_table
        F_value = aov_table.loc["session", "F Value"]
        p_value = aov_table.loc["session", "Pr > F"]
        excel_rows.append({"type":"nodal","region":i+1,"F_value":F_value,"p_value":p_value})

# Excel-Tabelle speichern
anova_xlsx = os.path.join(results_dir, "anova_results.xlsx")
pd.DataFrame(excel_rows).to_excel(anova_xlsx, index=False)
print(f"✅ ANOVA-Ergebnisse gespeichert als TXT: {anova_txt}")
print(f"✅ ANOVA-Ergebnisse gespeichert als XLSX: {anova_xlsx}")
