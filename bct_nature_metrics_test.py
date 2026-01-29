import os
import glob
import numpy as np
import pandas as pd
import bct

# -------------------------------
# Pfade & Sessions
# -------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(script_dir, "Test_matrizen")
sessions = ["ses-1", "ses-2", "ses-3", "ses-4"]

results_dir = os.path.join(script_dir, "results")
os.makedirs(results_dir, exist_ok=True)

# -------------------------------
# Matrix-Typ (nur BU / WU relevant)
# -------------------------------
def detect_matrix_type(A):
    if not np.allclose(A, A.T):
        return "unsupported"
    if np.any(A != A.astype(bool)):
        return "WU"
    else:
        return "BU"

# -------------------------------
# 12 Metrics nach Mousley et al.
# -------------------------------
def calculate_12_metrics(A, matrix_type):

    metrics = {}
    A = np.array(A, dtype=float)
    A_bin = (A > 0).astype(int)

    # ---------- GLOBAL ----------
    metrics["density"] = bct.density_und(A_bin)

    # Modularity
    Ci, Q = bct.modularity_louvain_und(A)
    metrics["modularity_Q"] = Q

    # Distances
    D = bct.distance_wei(A)

    # Global efficiency
    metrics["global_efficiency"] = bct.efficiency_wei(A, local=False)

    # Characteristic path length
    lambda_, efficiency, ecc, radius, diameter = bct.charpath(D)
    metrics["char_path_length"] = lambda_

    # Small-worldness
    metrics["small_worldness"] = bct.smallworldness(A)

    # Core / Periphery (Approximation)
    metrics["rich_club_coeff"] = np.nanmean(bct.rich_club_wu(A))

    # k-core (binär)
    metrics["k_core"] = np.max(bct.kcore_bu(A_bin))

    # ---------- LOCAL ----------
    metrics["degree_vec"] = bct.degrees_und(A_bin)
    metrics["strength_vec"] = bct.strengths_und(A)

    metrics["local_efficiency_vec"] = bct.efficiency_wei(A, local=True)
    metrics["clustering_vec"] = bct.clustering_coef_wu(A)

    metrics["betweenness_vec"] = bct.betweenness_wei(D)
    metrics["subgraph_centrality_vec"] = bct.subgraph_centrality(A)

    return metrics

# -------------------------------
# Schleife über Sessions & Files
# -------------------------------
all_data = []

for ses in sessions:
    ses_folder = os.path.join(root, ses)
    if not os.path.isdir(ses_folder):
        continue

    files = glob.glob(os.path.join(ses_folder, "*.npy"))
    for f in files:
        subject = os.path.splitext(os.path.basename(f))[0].split("_")[0]
        A = np.load(f)

        matrix_type = detect_matrix_type(A)
        if matrix_type == "unsupported":
            continue

        metrics = calculate_12_metrics(A, matrix_type)

        row = {
            "subject": subject,
            "session": ses,
            "matrix_type": matrix_type
        }

        for k, v in metrics.items():
            if isinstance(v, (np.ndarray, list)):
                for i, val in enumerate(v):
                    row[f"{k}_region_{i+1}"] = val
            else:
                row[k] = v

        all_data.append(row)

# -------------------------------
# Speichern
# -------------------------------
df = pd.DataFrame(all_data)
outfile = os.path.join(results_dir, "bct_mousley_12metrics.xlsx")
df.to_excel(outfile, index=False)

print(f"✔ Ergebnisse gespeichert: {outfile}")
