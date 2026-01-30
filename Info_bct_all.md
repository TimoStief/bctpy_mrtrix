# bct_all.py

## Beschreibung
Dieses Skript berechnet eine Vielzahl von **graphentheoretischen Netzwerkmetriken** für NPY-Matrizen
(z. B. aus dem Brainnectome-Atlas). Es unterstützt ungerichtete/gerichtete und gewichtete/ungewichtete
Matrizen:

- BU: Binary Undirected
- WU: Weighted Undirected
- BD: Binary Directed
- WD: Weighted Directed

Die folgenden Metriken werden berechnet:

- **Degree / Strength**
- **Density**
- **Clustering / Transitivity**
- **Efficiency** (binär/gewichtet; bei isolierten Knoten wird `NaN` gesetzt)
- **Components**
- **Community / Modularity (Louvain)**
- **Rich Club**
- **K-core**
- **Paths & Distances**
- **Betweenness (nur kleine Graphen ≤100 Knoten)**
- **Subgraph Centrality** (immer deaktiviert)

## Hinweise
- RuntimeWarnings bei der Berechnung von Efficiency sind normal, wenn die Matrix isolierte Knoten enthält.
- Skript flachst nodale Metriken in separate Spalten für Excel-Ausgabe.
- Unterstützt Ordnerstruktur mit Sessions: `npy_root/<session>/<subject>.npy`.

## Ausgabe
- Excel-Datei: `bct_all_metrics_brainnectome.xlsx`  
  Enthält alle globalen und nodalen Metriken für alle Subjects und Sessions.

## Verwendung
```python
# Beispiel:
import numpy as np
from bct_all import calculate_all_bct_metrics, detect_matrix_type

A = np.load("sub-001.npy")
matrix_type = detect_matrix_type(A)
metrics = calculate_all_bct_metrics(A, matrix_type)
