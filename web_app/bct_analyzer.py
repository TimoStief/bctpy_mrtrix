"""
Unified Brain Connectivity Toolbox Analysis Module
Combines functionality from bct_test.py, bct_all_test.py, and xlsx_to_npy.py
"""

import os
import glob
import numpy as np
import pandas as pd
import bct
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


class BCTAnalyzer:
    """Unified BCT analysis engine"""
    
    def __init__(self, output_callback=None, selected_metrics=None):
        """
        Initialize BCT Analyzer
        
        Args:
            output_callback: Optional callback function for logging output
            selected_metrics: List of metric groups to calculate (e.g., ['degree', 'clustering', 'efficiency'])
                            If None or empty, calculate all metrics
        """
        self.output_callback = output_callback or self._default_log
        self.sessions = ["ses-1", "ses-2", "ses-3", "ses-4"]
        
        # Map metric names to calculation flags
        self.metric_groups = {
            'degree': ['degree_vec', 'avg_degree', 'strength_vec', 'avg_strength'],
            'density': ['density'],
            'clustering': ['clustering', 'avg_clustering', 'transitivity', 'components', 'modularity'],
            'modularity': ['modularity'],
            'efficiency': ['efficiency', 'global_efficiency'],
            'pathlength': ['char_path_length'],
            'betweenness': ['betweenness_centrality', 'avg_betweenness'],
            'eigenvector': ['eigenvector_centrality', 'avg_eigenvector_centrality'],
            'assortativity': ['assortativity'],
            'qc': []  # QC metrics are always calculated
        }
        
        # If no metrics selected, default to all
        if not selected_metrics:
            self.selected_metrics = list(self.metric_groups.keys())
        else:
            self.selected_metrics = selected_metrics
        
    def _default_log(self, message: str):
        """Default logging function"""
        print(message)
    
    def log(self, message: str):
        """Log message via callback"""
        self.output_callback(message)
    
    def quality_check(self, A: np.ndarray, filename: str = "") -> Dict:
        """
        Perform quality checks on connectivity matrix
        
        Args:
            A: Connectivity matrix
            filename: Name of file being checked
        
        Returns:
            Dictionary with quality check results
        """
        qc = {
            "filename": filename,
            "passed": True,
            "warnings": [],
            "errors": []
        }
        
        # Check: Square matrix
        if A.shape[0] != A.shape[1]:
            qc["errors"].append(f"Not square: shape {A.shape}")
            qc["passed"] = False
            return qc
        
        n_nodes = A.shape[0]
        qc["n_nodes"] = n_nodes
        
        # Detect atlas from matrix size (common brain atlases)
        atlas_map = {
            78: "Brodmann (78 regions)",
            82: "AAL (82 regions)",
            90: "AAL90",
            116: "AAL116",
            200: "Schaefer200",
            246: "Brainnetome (246 regions)",
            264: "Power264",
            300: "Schaefer300",
            360: "HCP360",
            400: "Schaefer400"
        }
        qc["likely_atlas"] = atlas_map.get(n_nodes, f"Unknown ({n_nodes} regions)")
        
        # Check: Diagonal should be zero (no self-connections)
        diag_values = np.diag(A)
        if not np.allclose(diag_values, 0):
            nonzero_diag = np.sum(diag_values != 0)
            qc["warnings"].append(f"Diagonal has {nonzero_diag} non-zero values (should be 0)")
            qc["diagonal_mean"] = float(np.mean(diag_values))
            qc["diagonal_max"] = float(np.max(diag_values))
        else:
            qc["diagonal_ok"] = True
        
        # Check: Symmetry (upper vs lower triangle)
        is_symmetric = np.allclose(A, A.T)
        qc["is_symmetric"] = is_symmetric
        
        if is_symmetric:
            qc["matrix_type_hint"] = "Undirected (symmetric)"
        else:
            # Check degree of asymmetry
            upper_tri = np.triu(A, k=1)
            lower_tri = np.tril(A, k=-1).T
            diff = np.abs(upper_tri - lower_tri)
            asymmetry = float(np.mean(diff[upper_tri + lower_tri > 0]))
            qc["asymmetry_score"] = asymmetry
            qc["matrix_type_hint"] = "Directed (asymmetric)"
            if asymmetry > 0:
                qc["warnings"].append(f"Asymmetric matrix (avg difference: {asymmetry:.4f})")
        
        # Check: Sparsity and edge density
        if is_symmetric:
            # For symmetric matrices, only count upper triangle
            upper_tri_indices = np.triu_indices(n_nodes, k=1)
            nonzero_count = np.count_nonzero(A[upper_tri_indices])
            max_edges = n_nodes * (n_nodes - 1) // 2
        else:
            # For directed matrices, exclude diagonal
            np.fill_diagonal(A, 0)  # temporarily zero diagonal
            nonzero_count = np.count_nonzero(A)
            max_edges = n_nodes * (n_nodes - 1)
        
        sparsity = 1 - (nonzero_count / max_edges)
        qc["sparsity"] = float(sparsity)
        qc["edge_count"] = int(nonzero_count)
        qc["edge_density"] = float(1 - sparsity)
        
        if sparsity > 0.9:
            qc["warnings"].append(f"Very sparse matrix ({sparsity*100:.1f}% zeros)")
        elif sparsity < 0.1:
            qc["warnings"].append(f"Very dense matrix (only {sparsity*100:.1f}% zeros)")
        
        # Check: Weight distribution
        nonzero_weights = A[A > 0]
        if len(nonzero_weights) > 0:
            qc["weight_min"] = float(np.min(nonzero_weights))
            qc["weight_max"] = float(np.max(nonzero_weights))
            qc["weight_mean"] = float(np.mean(nonzero_weights))
            qc["weight_std"] = float(np.std(nonzero_weights))
            qc["weight_median"] = float(np.median(nonzero_weights))
            
            # Check for suspiciously low weights
            if qc["weight_min"] < 1e-10:
                qc["warnings"].append(f"Very small weights detected (min: {qc['weight_min']:.2e})")
            
            # Check for huge dynamic range
            if qc["weight_max"] / qc["weight_min"] > 1e6:
                qc["warnings"].append(f"Huge dynamic range in weights ({qc['weight_max']/qc['weight_min']:.2e})")
        else:
            qc["errors"].append("Matrix has no connections (all zeros)")
            qc["passed"] = False
        
        # Check: Missing or isolated nodes
        if is_symmetric:
            node_connections = np.sum(A > 0, axis=1)
        else:
            node_connections = np.sum(A > 0, axis=1) + np.sum(A > 0, axis=0)
        
        isolated_nodes = np.sum(node_connections == 0)
        if isolated_nodes > 0:
            qc["warnings"].append(f"{isolated_nodes} isolated nodes (no connections)")
            qc["isolated_nodes"] = int(isolated_nodes)
        
        weakly_connected = np.sum(node_connections < 3)
        if weakly_connected > n_nodes * 0.1:  # >10% of nodes
            qc["warnings"].append(f"{weakly_connected} weakly connected nodes (<3 connections)")
        
        # Check: NaN or Inf values
        if np.any(np.isnan(A)):
            qc["errors"].append("Matrix contains NaN values")
            qc["passed"] = False
        if np.any(np.isinf(A)):
            qc["errors"].append("Matrix contains Inf values")
            qc["passed"] = False
        
        # Check: Negative weights
        if np.any(A < 0):
            neg_count = np.sum(A < 0)
            qc["warnings"].append(f"{neg_count} negative weights (unusual for structural connectivity)")
            qc["negative_count"] = int(neg_count)
        
        return qc
    
    @staticmethod
    def detect_matrix_type(A):
        """
        Detect the type of connectivity matrix
        
        Returns:
            str: "BU" (binary undirected), "WU" (weighted undirected),
                 "BD" (binary directed), "WD" (weighted directed), or "unknown"
        """
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
    
    def calculate_all_metrics(self, A: np.ndarray, matrix_type: str) -> Dict:
        """
        Calculate selected BCT metrics for a connectivity matrix
        
        Args:
            A: Connectivity matrix
            matrix_type: Type of matrix (BU, WU, BD, WD)
        
        Returns:
            Dictionary of calculated metrics
        """
        metrics = {}
        
        # Helper function to check if metric should be calculated
        def should_calculate(metric_name):
            for group in self.selected_metrics:
                if metric_name in self.metric_groups.get(group, []):
                    return True
            return False
        
        # Ensure A is float
        A = np.array(A, dtype=float)
        A_bin = (A > 0).astype(int)
        
        # DEGREE & STRENGTH
        if 'degree' in self.selected_metrics:
            try:
                if matrix_type in ["BU", "WU"]:
                    metrics["degree_vec"] = bct.degrees_und(A_bin)
                    metrics["avg_degree"] = float(np.mean(metrics["degree_vec"]))
                    if matrix_type == "WU":
                        metrics["strength_vec"] = bct.strengths_und(A)
                        metrics["avg_strength"] = float(np.mean(metrics["strength_vec"]))
                elif matrix_type in ["BD", "WD"]:
                    in_deg, out_deg = bct.degrees_dir(A_bin, inout='all')
                    metrics["in_degree_vec"] = in_deg
                    metrics["out_degree_vec"] = out_deg
                    metrics["avg_in_degree"] = float(np.mean(in_deg))
                    metrics["avg_out_degree"] = float(np.mean(out_deg))
                    if matrix_type == "WD":
                        in_str, out_str = bct.strengths_dir(A, inout='all')
                        metrics["in_strength_vec"] = in_str
                        metrics["out_strength_vec"] = out_str
                        metrics["avg_in_strength"] = float(np.mean(in_str))
                        metrics["avg_out_strength"] = float(np.mean(out_str))
            except Exception as e:
                self.log(f"⚠️ Degree/Strength calculation failed: {e}")
        
        # DENSITY
        if 'density' in self.selected_metrics:
            try:
                if matrix_type in ["BU", "WU"]:
                    density_result = bct.density_und(A if matrix_type == "WU" else A_bin)
                    metrics["density"] = float(density_result[0]) if isinstance(density_result, tuple) else float(density_result)
                else:
                    density_result = bct.density_dir(A if matrix_type == "WD" else A_bin)
                    metrics["density"] = float(density_result[0]) if isinstance(density_result, tuple) else float(density_result)
            except Exception as e:
                self.log(f"⚠️ Density calculation failed: {e}")
        
        # CLUSTERING & COMMUNITY
        if 'clustering' in self.selected_metrics or 'modularity' in self.selected_metrics:
            try:
                if matrix_type == "BU":
                    clustering = bct.clustering_coef_bu(A_bin)
                    metrics["clustering"] = clustering[0] if isinstance(clustering, tuple) else clustering
                    metrics["avg_clustering"] = float(np.mean(metrics["clustering"]))
                    trans = bct.transitivity_bu(A_bin)
                    metrics["transitivity"] = float(trans[0]) if isinstance(trans, tuple) else float(trans)
                    eff = bct.efficiency_bin(A_bin)
                    metrics["efficiency"] = float(eff[0]) if isinstance(eff, tuple) else float(eff)
                    components = bct.get_components(A_bin)
                    metrics["components"] = len(components[0]) if isinstance(components, tuple) else len(components)
                    try:
                        Ci, Q = bct.community_louvain(A_bin)
                        metrics["modularity"] = float(Q)
                    except:
                        metrics["modularity"] = np.nan
                elif matrix_type == "WU":
                    clustering = bct.clustering_coef_wu(A)
                    metrics["clustering"] = clustering[0] if isinstance(clustering, tuple) else clustering
                    metrics["avg_clustering"] = float(np.mean(metrics["clustering"]))
                    trans = bct.transitivity_wu(A)
                    metrics["transitivity"] = float(trans[0]) if isinstance(trans, tuple) else float(trans)
                    eff = bct.efficiency_wei(A)
                    metrics["efficiency"] = float(eff[0]) if isinstance(eff, tuple) else float(eff)
                elif matrix_type == "BD":
                    clustering = bct.clustering_coef_bd(A_bin)
                    metrics["clustering"] = clustering[0] if isinstance(clustering, tuple) else clustering
                    metrics["avg_clustering"] = float(np.mean(metrics["clustering"]))
                    trans = bct.transitivity_bd(A_bin)
                    metrics["transitivity"] = float(trans[0]) if isinstance(trans, tuple) else float(trans)
                    eff = bct.efficiency_bin(A_bin)
                    metrics["efficiency"] = float(eff[0]) if isinstance(eff, tuple) else float(eff)
                elif matrix_type == "WD":
                    clustering = bct.clustering_coef_wd(A)
                    metrics["clustering"] = clustering[0] if isinstance(clustering, tuple) else clustering
                    metrics["avg_clustering"] = float(np.mean(metrics["clustering"]))
            except Exception as e:
                self.log(f"⚠️ Clustering calculation failed: {e}")
        
        # CENTRALITY METRICS
        if 'betweenness' in self.selected_metrics or 'eigenvector' in self.selected_metrics:
            try:
                if 'betweenness' in self.selected_metrics:
                    if matrix_type in ["BU", "BD"]:
                        bc = bct.betweenness_bin(A_bin)
                        metrics["betweenness_centrality"] = bc[0] if isinstance(bc, tuple) else bc
                        metrics["avg_betweenness"] = float(np.mean(metrics["betweenness_centrality"]))
                    elif matrix_type in ["WU", "WD"]:
                        bc = bct.betweenness_wei(A)
                        metrics["betweenness_centrality"] = bc[0] if isinstance(bc, tuple) else bc
                        metrics["avg_betweenness"] = float(np.mean(metrics["betweenness_centrality"]))
                
                # Eigenvector centrality (undirected only)
                if 'eigenvector' in self.selected_metrics and matrix_type in ["BU", "WU"]:
                    try:
                        ec = bct.eigenvector_centrality_und(A if matrix_type == "WU" else A_bin)
                        metrics["eigenvector_centrality"] = ec[0] if isinstance(ec, tuple) else ec
                        metrics["avg_eigenvector_centrality"] = float(np.mean(metrics["eigenvector_centrality"]))
                    except:
                        pass  # May fail for disconnected graphs
            except Exception as e:
                self.log(f"⚠️ Centrality calculation failed: {e}")
        
        # ASSORTATIVITY
        if 'assortativity' in self.selected_metrics:
            try:
                if matrix_type == "BU":
                    assort = bct.assortativity_bin(A_bin, flag=0)
                    metrics["assortativity"] = float(assort[0]) if isinstance(assort, tuple) else float(assort)
                elif matrix_type == "WU":
                    assort = bct.assortativity_wei(A, flag=0)
                    metrics["assortativity"] = float(assort[0]) if isinstance(assort, tuple) else float(assort)
            except Exception as e:
                self.log(f"⚠️ Assortativity calculation failed: {e}")
        
        # EFFICIENCY & PATH LENGTH
        if 'efficiency' in self.selected_metrics or 'pathlength' in self.selected_metrics:
            try:
                if matrix_type in ["BU", "WU"]:
                    # Characteristic path length
                    if matrix_type == "BU":
                        D = bct.distance_bin(A_bin)
                    else:
                        D = bct.distance_wei(A)
                    
                    D_result = D[0] if isinstance(D, tuple) else D
                    charpath_result = bct.charpath(D_result)
                    
                    if isinstance(charpath_result, tuple):
                        lambda_actual = charpath_result[0]  # characteristic path length
                    else:
                        lambda_actual = charpath_result
                    
                    metrics["char_path_length"] = float(lambda_actual)
                    
                    # Global efficiency (already calculated in clustering section)
                    if 'efficiency' in metrics:
                        metrics["global_efficiency"] = float(metrics.get("efficiency", 0))
            except Exception as e:
                self.log(f"⚠️ Path length calculation failed: {e}")
        
        return metrics
    
    def analyze_matrices(self, input_dir: str, output_dir: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Analyze all matrices in input directory
        
        Args:
            input_dir: Directory containing session folders with .npy files
            output_dir: Directory for output files
        
        Returns:
            Tuple of (results_dataframe, metrics_summary)
        """
        self.log(f"📁 Input directory: {input_dir}")
        self.log(f"📁 Output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        all_data = []
        metrics_summary = {}
        
        # Check for session folders
        found_sessions = [s for s in self.sessions if os.path.isdir(os.path.join(input_dir, s))]
        if not found_sessions:
            self.log("⚠️ No session folders found (ses-1, ses-2, ses-3, ses-4)")
            return pd.DataFrame(), metrics_summary
        
        self.log(f"✓ Found sessions: {', '.join(found_sessions)}\n")
        
        # Quality control summary
        qc_summary = {
            "total_files": 0,
            "passed_qc": 0,
            "warnings": 0,
            "errors": 0,
            "atlases_detected": {}
        }
        
        for session in found_sessions:
            session_path = os.path.join(input_dir, session)
            npy_files = glob.glob(os.path.join(session_path, "*.npy"))
            
            if not npy_files:
                self.log(f"⚠️ No .npy files found in {session}")
                continue
            
            self.log(f"📊 Processing {session}: {len(npy_files)} file(s)")
            
            for npy_file in npy_files:
                try:
                    filename = os.path.basename(npy_file)
                    self.log(f"  → {filename}")
                    
                    # Load matrix
                    A = np.load(npy_file)
                    
                    # QUALITY CONTROL
                    qc_result = self.quality_check(A, filename)
                    qc_summary["total_files"] += 1
                    
                    # Log QC results
                    if qc_result["passed"]:
                        qc_summary["passed_qc"] += 1
                        self.log(f"    ✓ QC passed: {qc_result['likely_atlas']}")
                    else:
                        self.log(f"    ❌ QC failed")
                        for error in qc_result["errors"]:
                            self.log(f"       Error: {error}")
                    
                    # Log warnings
                    if qc_result["warnings"]:
                        qc_summary["warnings"] += len(qc_result["warnings"])
                        for warning in qc_result["warnings"]:
                            self.log(f"    ⚠️ {warning}")
                    
                    # Track atlas usage
                    atlas = qc_result.get("likely_atlas", "Unknown")
                    qc_summary["atlases_detected"][atlas] = qc_summary["atlases_detected"].get(atlas, 0) + 1
                    
                    matrix_type = self.detect_matrix_type(A)
                    self.log(f"    Matrix type: {matrix_type}")
                    
                    # Calculate metrics
                    metrics = self.calculate_all_metrics(A, matrix_type)
                    
                    # Extract subject name
                    subject = filename.split("_")[0]
                    
                    # Create data row
                    data_row = {
                        "subject": subject,
                        "session": session,
                        "filename": filename,
                        "matrix_type": matrix_type,
                        "matrix_shape": str(A.shape),
                        "qc_passed": qc_result["passed"],
                        "qc_warnings": len(qc_result["warnings"]),
                        "atlas": qc_result.get("likely_atlas", "Unknown"),
                        "n_nodes": qc_result.get("n_nodes", A.shape[0]),
                        "sparsity": qc_result.get("sparsity", np.nan),
                        "edge_density": qc_result.get("edge_density", np.nan),
                    }
                    
                    # Add QC metrics if available
                    for qc_key in ["diagonal_ok", "is_symmetric", "isolated_nodes", 
                                   "weight_mean", "weight_std", "weight_min", "weight_max"]:
                        if qc_key in qc_result:
                            data_row[f"qc_{qc_key}"] = qc_result[qc_key]
                    
                    # Add metrics to row
                    for key, value in metrics.items():
                        if isinstance(value, np.ndarray):
                            data_row[f"{key}_mean"] = float(np.mean(value))
                        else:
                            data_row[key] = value
                    
                    all_data.append(data_row)
                    
                except Exception as e:
                    self.log(f"  ❌ Error processing {filename}: {e}")
        
        # Create results dataframe
        if all_data:
            df_results = pd.DataFrame(all_data)
            self.log(f"\n✓ Processed {len(all_data)} matrices")
            
            # Save results
            output_xlsx = os.path.join(output_dir, "bct_analysis_results.xlsx")
            df_results.to_excel(output_xlsx, index=False)
            self.log(f"✓ Results saved to: {output_xlsx}")
            
            # Save QC report
            qc_report_path = os.path.join(output_dir, "quality_control_report.txt")
            with open(qc_report_path, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("QUALITY CONTROL REPORT\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Total files processed: {qc_summary['total_files']}\n")
                f.write(f"Passed QC: {qc_summary['passed_qc']}\n")
                f.write(f"Total warnings: {qc_summary['warnings']}\n")
                f.write(f"Total errors: {qc_summary['errors']}\n\n")
                f.write("Brain Atlases Detected:\n")
                for atlas, count in qc_summary['atlases_detected'].items():
                    f.write(f"  - {atlas}: {count} files\n")
                f.write("\n" + "=" * 60 + "\n")
            
            self.log(f"✓ QC report saved to: {qc_report_path}")
            
            # Create summary with JSON-serializable types
            matrix_counts = {
                key: int(val) for key, val in df_results["matrix_type"].value_counts().to_dict().items()
            }

            metrics_summary = {
                "total_matrices": int(len(all_data)),
                "sessions_processed": [str(s) for s in found_sessions],
                "subjects": [str(s) for s in df_results["subject"].unique().tolist()],
                "matrix_types": matrix_counts,
                "quality_control": qc_summary
            }
            
            return df_results, metrics_summary
        else:
            self.log("❌ No matrices were successfully processed")
            return pd.DataFrame(), metrics_summary
    
    def convert_xlsx_to_npy(self, input_xlsx: str, output_dir: str) -> int:
        """
        Convert Excel metrics to NPY files
        
        Args:
            input_xlsx: Path to Excel file
            output_dir: Directory for output NPY files
        
        Returns:
            Number of files created
        """
        self.log(f"📄 Loading Excel file: {input_xlsx}")
        os.makedirs(output_dir, exist_ok=True)
        
        df = pd.read_excel(input_xlsx)
        self.log(f"✓ Loaded {len(df)} rows from Excel")
        
        count = 0
        for idx, row in df.iterrows():
            try:
                # Extract vector columns
                vector_cols = [c for c in df.columns if '_region_' in c or '_vec' in c]
                if not vector_cols:
                    continue
                
                vectors = row[vector_cols].values.astype(float)
                
                subject = row.get('subject', f'subject_{idx}')
                session = row.get('session', 'unknown')
                
                filename = f"{subject}_{session}_metrics.npy"
                output_path = os.path.join(output_dir, filename)
                
                np.save(output_path, vectors)
                self.log(f"✓ Saved: {filename}")
                count += 1
            except Exception as e:
                self.log(f"❌ Error processing row {idx}: {e}")
        
        self.log(f"✓ Conversion complete: {count} files created")
        return count
