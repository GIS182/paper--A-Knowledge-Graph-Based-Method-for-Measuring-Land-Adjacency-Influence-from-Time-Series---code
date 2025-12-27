[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18061451.svg)](https://doi.org/10.5281/zenodo.18061451)


# A Knowledge Graph-Based Method for Measuring Land Adjacency Influence from Time-Series Land-Use Data

This repository provides the **official implementation and experimental data**
for the paper:

> **A Knowledge Graph-Based Method for Measuring Land Adjacency Influence from Time-Series Land-Use Data**

The code implements a structured framework for quantifying **land-use adjacency influence**
by integrating spatial proximity relationships and temporal coupling patterns
derived from multi-year land-use data.

The proposed method constructs two core knowledge graphs:

- **Land-use Adjacency Distance Map (LAIM)** — modeling spatial adjacency distance relations
- **Land-use Coupling Strength Map (LCSM)** — modeling temporal evolution coupling

These graphs are further integrated to compute the **Land-use Adjacency Influence Index (LAI_Index)**,
which can be used as an interpretable quantitative factor in land-use analysis
and ecological sensitivity assessment.

---

## 1. Repository Structure

```
LandAdjacency_Serial/
├── core/ # Core algorithm modules
│ ├── preprocess.py # Data preprocessing
│ ├── extract_features.py # Spatial and temporal feature extraction
│ ├── adjacency_detector.py # Adjacency relationship detection
│ ├── build_laim.py # Construction of LAIM
│ ├── build_lcsm.py # Construction of LCSM
│ ├── compute_effect.py # LAI_Index computation
│ ├── visualize_matrix.py # Visualization utilities
│ └── output/ # Generated matrices and figures
│
├── utils/ # Utility functions
│ ├── geodata_io.py
│ ├── spatial_index.py
│ ├── config_reader.py
│ └── timer.py
│
├── config/ # Configuration files
│ ├── config.yaml # Global experiment settings
│ ├── class_map.json # Land-use category mapping
│ └── weight_config.json # Weight and decay parameters
│
├── data/ # Input land-use datasets
│ ├── laim_matrix.csv # Output LAIM adjacency matrix
│ ├── lcsm_matrix.csv # Output LCSM coupling matrix
│ └── xfdnlanduse.gpkg # Integrated GeoPackage
│
├── main.py # Main execution script
├── requirements.txt # Python dependencies
└── README.md
```
---

## 2. Data Description

- **Data type**: Multi-year land-use vector data (polygon-based)
- **Temporal coverage**: 2014–2024
- **Spatial unit**: Land-use patches
- **Coordinate system**: Projected coordinate system (consistent across years)

The dataset is organized chronologically and used to extract:

- Spatial adjacency distance relationships (for LAIM)
- Temporal land-use transition coupling (for LCSM)

All data required to reproduce the main experiments reported in the paper
are included in this repository.

---

## 3. Methodological Overview

The proposed framework consists of three main stages:

1. **Spatial Adjacency Modeling**
   - Identification of neighboring land-use patches
   - Distance-based decay modeling
   - Construction of the Land-use Adjacency Distance Map (LAIM)

2. **Temporal Coupling Modeling**
   - Extraction of inter-annual land-use transitions
   - Quantification of coupling strength across time
   - Construction of the Land-use Coupling Strength Map (LCSM)

3. **Adjacency Influence Quantification**
   - Integration of LAIM and LCSM
   - Computation of the Land-use Adjacency Influence Index (LAI_Index)
   - Generation of adjacency influence matrices and visualizations

---

## 4. Environment Setup

### 4.1 Requirements

- Python ≥ 3.9
- Recommended OS: Windows / Linux

Install dependencies via:

```bash
pip install -r requirements.txt
```

Key libraries include:

- `geopandas`
- `shapely`
- `rasterio`
- `numpy`
- `pandas`
- `networkx`
- `matplotlib`

------

## 5. Running the Code

To reproduce the main experimental results:

```bash
python main.py
```

The script will sequentially:

1. Preprocess land-use data
2. Construct LAIM and LCSM
3. Compute the LAI_Index
4. Export adjacency matrices and visual outputs

Results will be saved in:

```
core/output/
```

------

## 6. Outputs

Main outputs include:

- `laim_matrix.csv`: Spatial adjacency distance matrix
- `lcsm_matrix.csv`: Temporal coupling strength matrix
- Heatmap visualizations of LAIM and LCSM
- Intermediate knowledge graph representations

These outputs correspond directly to the quantitative analyses
presented in the paper.

------

## 7. Reproducibility and Notes

- All parameters used in the experiments are explicitly defined
  in the configuration files under `config/`.
- The framework avoids resolution resampling to preserve
  patch geometry and adjacency integrity.
- Results may vary slightly if alternative spatial distance metrics
  or temporal windows are adopted.

------

## 8. Citation

If you use this code or data in your research, please cite the paper:

```
[To be updated after publication]
```

------

## 9. License

This project is released for **academic research purposes only**.
Please contact the authors for commercial use.

------

## 10. Contact

For questions or reproducibility issues, please contact the corresponding author.
