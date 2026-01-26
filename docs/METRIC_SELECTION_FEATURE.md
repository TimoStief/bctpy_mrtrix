# Metric Selection Feature - Implementation Summary

## Overview
The web interface now includes a **grouped metric selection panel** with checkboxes to include/exclude metric calculations. This allows users to customize their analysis and reduce computation time for large datasets.

## UI Components

### Metric Groups (6 categories)

1. **Basic Properties**
   - ✓ Degree/Strength (default: checked)
   - ✓ Density/Sparsity (default: checked)

2. **Clustering & Community**
   - ✓ Clustering Coefficient (default: checked)
   - ✓ Modularity/Community (default: checked)

3. **Integration & Efficiency**
   - ✓ Global Efficiency (default: checked)
   - ✓ Characteristic Path Length (default: checked)

4. **Centrality Measures**
   - ✓ Betweenness Centrality (default: checked)
   - ✓ Eigenvector Centrality (default: unchecked)

5. **Network Properties**
   - ✓ Assortativity (default: unchecked)

6. **Quality Control**
   - ✓ QC Metrics & Atlas Detection (default: checked - always includes QC)

## Technical Implementation

### Frontend (JavaScript)
- Added metric selection panel with grouped checkboxes
- Collects selected metrics on analysis start
- Sends `selected_metrics` array to backend API

### Backend (Flask)
- `/api/analyze` endpoint now accepts `selected_metrics` parameter
- Passes metric selections to BCTAnalyzer initialization
- Logs selected metrics to terminal output

### Analyzer Core (bct_analyzer.py)
- `__init__` accepts optional `selected_metrics` parameter
- Maps metric names to internal calculation flags
- `calculate_all_metrics()` conditionally calculates only selected metrics
- Reduces computation time for unselected metrics

## Default Configuration

### Always Calculated
- Quality Control (diagonal check, symmetry, atlas, weight stats, isolated nodes, etc.)
- Basic properties (degree/strength, density)
- Clustering metrics

### Optional Metrics
- **Eigenvector Centrality**: Unchecked by default (can fail on disconnected graphs)
- **Assortativity**: Unchecked by default (optional advanced metric)

## User Experience

1. User selects input/output directories
2. User optionally customizes metric selection
3. Each metric group can be toggled independently
4. Selected metrics displayed in terminal output: `📊 Selected metrics: degree, clustering, efficiency`
5. Only selected metrics appear in output Excel file
6. QC report always generated regardless of selections

## Data Output

### Excel File (`bct_analysis_results.xlsx`)
- Contains only selected metric columns (plus QC columns)
- Reduces file size when metrics are excluded
- Per-node metric vectors excluded if not needed

### Quality Control Report (`quality_control_report.txt`)
- Always generated with full QC summary
- Independent of metric selections

## Performance Benefits

- **Skip expensive calculations**: Eigenvector centrality can be slow on large networks
- **Reduce output size**: Only selected metrics stored in results
- **Faster execution**: Conditional metric calculation reduces runtime
- **Memory efficient**: Per-node vectors not stored if not needed

## Example Workflow

### Fast Analysis (Basic metrics only)
```
✓ Degree/Strength
✓ Density/Sparsity
✗ Clustering & Community
✗ Integration & Efficiency
✗ Centrality Measures
✓ Quality Control
```

### Comprehensive Analysis (All metrics)
```
✓ All groups selected
↓ Complete BCT analysis
↓ Longer runtime but comprehensive results
```

### Hub Analysis (Focus on connectivity)
```
✓ Degree/Strength
✓ Centrality Measures
✓ Quality Control
✗ Others
↓ Fast identification of network hubs
```

## Files Modified

1. **web_app/templates/index.html**
   - Added metric selection UI panel with checkboxes
   - Updated `startAnalysis()` to collect selected metrics

2. **web_app/app.py**
   - Updated `/api/analyze` to accept metric selections
   - Logs selected metrics to output

3. **web_app/bct_analyzer.py**
   - Added `selected_metrics` parameter to `__init__`
   - Added `metric_groups` mapping dictionary
   - Made `calculate_all_metrics()` conditional
   - All metric calculation blocks wrapped in condition checks

## Future Enhancements

- Save/load favorite metric configurations
- Quick presets (e.g., "Fast", "Complete", "Hub Analysis")
- Metric descriptions/tooltips on hover
- Performance estimates for selected metrics
- Automatic optimization suggestions based on network size
