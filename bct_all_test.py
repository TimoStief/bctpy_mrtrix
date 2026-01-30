import os
import glob
import numpy as np
import pandas as pd
import bct

# =========================
# XLSX → NPY Konvertierung
# =========================

input_dir = r"C:\Users\timo-\Desktop\Forschung\bctpy_mrtrix\Test_matrizen\brainnectome_count_matrizen"
npy_root  = os.path.join(input_dir, "npy_matrizen")
os.makedirs(npy_root, exist_ok=True)

# Alle XLSX-Dateien (rekursiv, aber NICHT im npy_root)
xlsx_files = [
    f for f in glob.glob(os.path.join(input_dir, "**", "*.xlsx"), recursive=True)
    if npy_root not in os.path.abspath(f)
]

# Alle vorhandenen NPYs sammeln (nur Basename ohne Endung!)
existing_npy_names = {
    os.path.splitext(os.path.basename(f))[0]
    for f in glob.glob(os.path.join(npy_root, "**", "*.npy"), recursive=True)
}

print(f"Gefundene XLSX: {len(xlsx_files)}")
print(f"Gefundene vorhandene NPYs: {len(existing_npy_names)}")

if not xlsx_files and not existing_npy_names:
    raise FileNotFoundError(
        f"Keine .xlsx oder .npy Dateien im Ordner {input_dir} gefunden"
    )

# -------------------------
# Konvertierung (mit Skip)
# -------------------------
for infile in xlsx_files:
    subject_name = os.path.splitext(os.path.basename(infile))[0]

    # 🔒 Globaler Skip-Check (unabhängig vom Ordner!)
    if subject_name in existing_npy_names:
        print(f"↪ Bereits konvertiert, übersprungen: {subject_name}")
        continue

    session_name = os.path.basename(os.path.dirname(infile))
    session_dir = os.path.join(npy_root, session_name)
    os.makedirs(session_dir, exist_ok=True)

    out_file = os.path.join(session_dir, f"{subject_name}.npy")

    print(f"▶ Konvertiere XLSX: {subject_name}")

    df = pd.read_excel(infile, header=None)
    matrix = df.values.astype(float)

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Matrix nicht quadratisch ({matrix.shape}): {infile}")

    np.save(out_file, matrix)
    existing_npy_names.add(subject_name)  # wichtig bei langen Läufen

    print(f"  ✔ gespeichert: {out_file}")

# -------------------------------
# Pfade & Sessions
# -------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(script_dir, "Test_matrizen/brainnectome_count_matrizen")
sessions = ["ses-1", "ses-2", "ses-3", "ses-4"]
results_dir = os.path.join(script_dir, "results")
os.makedirs(results_dir, exist_ok=True)

# -------------------------------
# Hilfsfunktion: Matrix-Typ bestimmen
# -------------------------------
def detect_matrix_type(A):
    if A.shape[0] != A.shape[1]:
        return "unknown"
    if np.allclose(A, A.T):
        if np.any(A != A.astype(bool)):
            return "WU"  # weighted undirected
        else:
            return "BU"  # binary undirected
    else:
        if np.any(A != A.astype(bool)):
            return "WD"  # weighted directed
        else:
            return "BD"  # binary directed
# -------------------------------
# Alle Metriken berechnen
# -------------------------------
def calculate_all_bct_metrics(A, matrix_type):
    metrics = {}

    # Sicherstellen, dass A float ist
    A = np.array(A, dtype=float)
    np.fill_diagonal(A, 0)  # 🔒 wichtig für Distance/Efficiency
    A_bin = (A > 0).astype(int)

    # Prüfen auf isolierte Knoten
    has_isolated_nodes = np.any(np.sum(A_bin, axis=0) == 0)

    # -------------------
    # DEGREE & STRENGTH
    # -------------------
    try:
        if matrix_type in ["BU", "WU"]:
            metrics["degree_vec"] = bct.degrees_und(A_bin)
            metrics["avg_degree"] = np.mean(metrics["degree_vec"])
            if matrix_type == "WU":
                metrics["strength_vec"] = bct.strengths_und(A)
                metrics["avg_strength"] = np.mean(metrics["strength_vec"])
        elif matrix_type in ["BD", "WD"]:
            in_deg, out_deg = bct.degrees_dir(A_bin, inout='all')
            metrics["in_degree_vec"] = in_deg
            metrics["out_degree_vec"] = out_deg
            if matrix_type == "WD":
                in_str, out_str = bct.strengths_dir(A, inout='all')
                metrics["in_strength_vec"] = in_str
                metrics["out_strength_vec"] = out_str
    except Exception as e:
        print(f"Degree/Strength konnte nicht berechnet werden: {e}")

    # -------------------
    # DENSITY
    # -------------------
    try:
        if matrix_type in ["BU", "WU"]:
            metrics["density"] = bct.density_und(A if matrix_type == "WU" else A_bin)
        else:
            metrics["density"] = bct.density_dir(A if matrix_type == "WD" else A_bin)
    except Exception as e:
        print(f"Density konnte nicht berechnet werden: {e}")

    # -------------------
    # CLUSTERING & COMMUNITY
    # -------------------
    try:
        if matrix_type == "BU":
            metrics["clustering"] = bct.clustering_coef_bu(A_bin)
            metrics["transitivity"] = bct.transitivity_bu(A_bin)

            # 🔒 Efficiency nur bei vollständig verbundenen Graphen
            metrics["efficiency"] = (
                np.nan if has_isolated_nodes else bct.efficiency_bin(A_bin)
            )

            metrics["components"] = len(bct.get_components(A_bin))

            try:
                Ci, Q = bct.community_louvain(A_bin)
                metrics["modularity"] = Q
                metrics["community_louvain"] = Ci
            except:
                metrics["modularity"] = np.nan
                metrics["community_louvain"] = None

            metrics["rich_club"] = bct.rich_club_bu(A_bin)
            metrics["kcore"] = bct.kcore_bu(A_bin)

        elif matrix_type == "WU":
            metrics["clustering"] = bct.clustering_coef_wu(A)
            metrics["transitivity"] = bct.transitivity_wu(A)

            metrics["efficiency"] = (
                np.nan if has_isolated_nodes else bct.efficiency_wei(A)
            )

        elif matrix_type == "BD":
            metrics["clustering"] = bct.clustering_coef_bd(A_bin)
            metrics["transitivity"] = bct.transitivity_bd(A_bin)

            metrics["efficiency"] = (
                np.nan if has_isolated_nodes else bct.efficiency_bin(A_bin)
            )

        elif matrix_type == "WD":
            metrics["clustering"] = bct.clustering_coef_wd(A)
            metrics["transitivity"] = bct.transitivity_wd(A)

            metrics["efficiency"] = (
                np.nan if has_isolated_nodes else bct.efficiency_wei(A)
            )

    except Exception as e:
        print(f"Clustering/Community konnte nicht berechnet werden: {e}")

    # -------------------
    # PATHS & DISTANCES (mit Safe-Check für isolierte Knoten)
    # -------------------
    try:
        if matrix_type in ["BU", "BD"]:
            # Prüfen, ob isolierte Knoten existieren
            if np.any(np.sum(A_bin, axis=0) == 0):
                metrics["distance_bin"] = np.nan
                metrics["charpath"] = np.nan
            else:
                metrics["distance_bin"] = bct.distance_bin(A_bin)
                metrics["charpath"] = bct.charpath(A_bin)
        else:
            # Für gewichtete Matrizen
            if np.any(np.sum(A, axis=0) == 0):
                metrics["distance_wei"] = np.nan
                metrics["charpath"] = np.nan
            else:
                metrics["distance_wei"] = bct.distance_wei(A)
                metrics["charpath"] = bct.charpath(A)
    except Exception as e:
        print(f"Paths/Distances konnte nicht berechnet werden: {e}")
    # -------------------
    # CENTRALITY (bewusst reduziert)
    # -------------------
    try:
        if matrix_type in ["BU", "BD"] and A_bin.shape[0] <= 100:
            metrics["betweenness_bin"] = bct.betweenness_bin(A_bin)
        else:
            metrics["betweenness_bin"] = np.nan

        metrics["subgraph_centrality"] = np.nan

    except Exception as e:
        print(f"Centrality übersprungen: {e}")

    return metrics


# -------------------------------
# Schleife über alle Sessions / Dateien (angepasst für npy_root)
# -------------------------------
all_data = []

for ses in sessions:
    ses_folder = os.path.join(npy_root, ses)  # <-- hier: npy_root statt root
    if not os.path.isdir(ses_folder):
        print(f"Session-Ordner existiert nicht: {ses_folder}")
        continue

    files = glob.glob(os.path.join(ses_folder, "*.npy"))
    if not files:
        print(f"Keine NPY-Dateien gefunden in {ses_folder}")
        continue

    for f in files:
        subject = os.path.splitext(os.path.basename(f))[0].split("_")[0]
        A = np.load(f)
        matrix_type = detect_matrix_type(A)
        metrics = calculate_all_bct_metrics(A, matrix_type)

        # Flatten nodale Vektoren in separate Spalten
        flat_metrics = {"subject": subject, "session": ses, "matrix_type": matrix_type}
        for k, v in metrics.items():
            if isinstance(v, (list, np.ndarray)):
                for i, val in enumerate(v):
                    flat_metrics[f"{k}_region_{i+1}"] = val
            else:
                flat_metrics[k] = v

        all_data.append(flat_metrics)

# -------------------------------
# Ergebnisse speichern
# -------------------------------
df = pd.DataFrame(all_data)
output_file = os.path.join(results_dir, "bct_all_metrics_brainnectome.xlsx")
df.to_excel(output_file, index=False)
print(f"Alle BCT-Metriken gespeichert in {output_file}")
