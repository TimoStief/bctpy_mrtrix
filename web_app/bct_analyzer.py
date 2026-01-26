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
    
    def __init__(self, output_callback=None):
        """
        Initialize BCT Analyzer
        
        Args:
            output_callback: Optional callback function for logging output
        """
        self.output_callback = output_callback or self._default_log
        self.sessions = ["ses-1", "ses-2", "ses-3", "ses-4"]
        
    def _default_log(self, message: str):
        """Default logging function"""
        print(message)
    
    def log(self, message: str):
        """Log message via callback"""
        self.output_callback(message)
    
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
        Calculate all BCT metrics for a connectivity matrix
        
        Args:
            A: Connectivity matrix
            matrix_type: Type of matrix (BU, WU, BD, WD)
        
        Returns:
            Dictionary of calculated metrics
        """
        metrics = {}
        
        # Ensure A is float
        A = np.array(A, dtype=float)
        A_bin = (A > 0).astype(int)
        
        # DEGREE & STRENGTH
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
        try:
            if matrix_type in ["BU", "WU"]:
                metrics["density"] = float(bct.density_und(A if matrix_type == "WU" else A_bin))
            else:
                metrics["density"] = float(bct.density_dir(A if matrix_type == "WD" else A_bin))
        except Exception as e:
            self.log(f"⚠️ Density calculation failed: {e}")
        
        # CLUSTERING & COMMUNITY
        try:
            if matrix_type == "BU":
                metrics["clustering"] = bct.clustering_coef_bu(A_bin)
                metrics["avg_clustering"] = float(np.mean(metrics["clustering"]))
                metrics["transitivity"] = float(bct.transitivity_bu(A_bin))
                metrics["efficiency"] = float(bct.efficiency_bin(A_bin))
                metrics["components"] = len(bct.get_components(A_bin))
                try:
                    Ci, Q = bct.community_louvain(A_bin)
                    metrics["modularity"] = float(Q)
                except:
                    metrics["modularity"] = np.nan
            elif matrix_type == "WU":
                metrics["clustering"] = bct.clustering_coef_wu(A)
                metrics["avg_clustering"] = float(np.mean(metrics["clustering"]))
                metrics["transitivity"] = float(bct.transitivity_wu(A))
                metrics["efficiency"] = float(bct.efficiency_wei(A))
            elif matrix_type == "BD":
                metrics["clustering"] = bct.clustering_coef_bd(A_bin)
                metrics["avg_clustering"] = float(np.mean(metrics["clustering"]))
                metrics["transitivity"] = float(bct.transitivity_bd(A_bin))
                metrics["efficiency"] = float(bct.efficiency_bin(A_bin))
            elif matrix_type == "WD":
                metrics["clustering"] = bct.clustering_coef_wd(A)
                metrics["avg_clustering"] = float(np.mean(metrics["clustering"]))
        except Exception as e:
            self.log(f"⚠️ Clustering calculation failed: {e}")
        
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
                    }
                    
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
            
            # Create summary
            metrics_summary = {
                "total_matrices": len(all_data),
                "sessions_processed": found_sessions,
                "subjects": list(df_results["subject"].unique()),
                "matrix_types": dict(df_results["matrix_type"].value_counts()),
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
