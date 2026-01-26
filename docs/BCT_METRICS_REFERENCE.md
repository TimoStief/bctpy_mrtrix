# BCT Metrics Reference

## Currently Implemented Metrics

### 1. Basic Properties
- **Degree/Strength**: Node connectivity measures
  - `degree_vec`: Number of connections per node
  - `avg_degree`: Average degree across network
  - `strength_vec`: Sum of edge weights per node (weighted networks)
  - `avg_strength`: Average strength across network

### 2. Density & Sparsity
- **Density**: Proportion of actual vs possible connections
- **Sparsity**: Proportion of missing connections

### 3. Clustering & Segregation
- **Clustering Coefficient**: Local connectivity (triangles around nodes)
  - `clustering`: Per-node clustering coefficients
  - `avg_clustering`: Network average
- **Transitivity**: Global clustering measure
- **Modularity**: Community structure strength (Q value)

### 4. Integration & Efficiency
- **Global Efficiency**: Network-wide information transfer
- **Characteristic Path Length**: Average shortest path length

### 5. Centrality Measures
- **Betweenness Centrality**: Nodes on shortest paths
  - `betweenness_centrality`: Per-node values
  - `avg_betweenness`: Network average
- **Eigenvector Centrality**: Influence based on connections to important nodes

### 6. Network Properties
- **Assortativity**: Tendency for similar nodes to connect
- **Components**: Number of connected subgraphs

## Quality Control Checks

### Automatically Detected
1. **Matrix Shape**: Square matrix validation
2. **Brain Atlas**: Detected from matrix dimensions
   - 78 regions → Brodmann
   - 82/90/116 → AAL variants
   - 200/300/400 → Schaefer variants
   - 246 → Brainnetome
   - 264 → Power264
   - 360 → HCP360

3. **Diagonal Values**: Should be zero (no self-connections)
4. **Symmetry**: Upper vs lower triangle consistency
5. **Sparsity**: Connection density analysis
6. **Weight Distribution**: Min/max/mean/median statistics
7. **Isolated Nodes**: Nodes with no connections
8. **Missing Data**: NaN or Inf values
9. **Negative Weights**: Unusual for structural connectivity
10. **Dynamic Range**: Very large weight variations

## Additional Available BCT Metrics (Not Yet Implemented)

### Path & Distance
- `charpath`: Characteristic path length with more details
- `distance_wei`: Weighted distance matrix
- `findpaths`: Find shortest paths
- `diffusion_efficiency`: Alternative efficiency measure

### Community Detection
- `modularity_louvain_und`: Louvain community detection
- `modularity_finetune_und`: Fine-tuned modularity
- `consensus_und`: Consensus community detection
- `partition_distance`: Compare community structures

### Rich Club & Hierarchy
- `rich_club_wu/bu`: Rich club coefficients
- `core_periphery_dir`: Core-periphery structure
- `kcoreness_centrality_bu`: K-core decomposition

### Motifs & Subgraphs
- `motif3struct_bin/wei`: 3-node motifs
- `motif4struct_bin/wei`: 4-node motifs
- `find_motif34`: Search specific motifs

### Resilience & Robustness
- `backbone_wu`: Network backbone
- `threshold_proportional`: Threshold to fixed density

### Small-World Metrics
- Path length ratio vs random
- Clustering ratio vs random
- Small-world coefficient (σ)

### Advanced Centrality
- `pagerank_centrality`: PageRank algorithm
- `subgraph_centrality`: Subgraph participation
- `module_degree_zscore`: Within-module z-score

### Network Generation
- Random networks for comparison
- Lattice networks
- Small-world networks (Watts-Strogatz)

## Output Files

### 1. `bct_analysis_results.xlsx`
Excel spreadsheet with one row per matrix containing:
- Subject ID, session, filename
- QC metrics (pass/fail, warnings)
- All calculated graph metrics
- Per-node metrics averaged

### 2. `quality_control_report.txt`
Text summary of QC results:
- Total files processed
- Pass/fail counts
- Atlas distribution
- Warning/error summaries

## Recommendations for Future Extensions

### Priority Additions
1. **Small-World Analysis**: Compare to random networks
2. **Rich Club Analysis**: Hub connectivity patterns
3. **Hub Detection**: Identify highly connected regions
4. **Community Structure**: Detailed module analysis
5. **Resilience Testing**: Attack/failure simulations

### Statistical Extensions
- Multi-session comparisons
- Group-level statistics
- Longitudinal changes
- Network-based statistics (NBS)

### Visualization
- Connectivity matrices (heatmaps)
- Network graphs (spring layouts)
- Brain surface projections
- Metric distributions
