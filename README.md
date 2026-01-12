# BreastCancerRaceDisparities  
*Tumor Microenvironment Analysis by Race*

This project focuses on analyzing the **tumor microenvironment (TME)** in breast cancer tissue and exploring its relationship with neighborhood deprivation and race. Using deep learning-derived tissue classifications, the repository extracts spatial, area, and adjacency features to study differences between Black and White patients, providing insights into potential health disparities.

---

## Features

- **Data Loading & Processing**: Reads deep learning-inferred tissue classifications from histology slides.  
- **Subsampling & Grid Construction**: Reduces large tissue maps into manageable grids for analysis.  
- **Delaunay Triangulation**: Visualizes spatial organization of tissue tiles.  
- **Feature Extraction**:
  - Tissue **class distribution** across slides  
  - **Spatial distribution** of tissue types  
  - **Area and shape** properties of tissues (size, perimeter)  
  - **Adjacency/co-occurrence measures** between tissue types  
- **Slide-wise Aggregation**: Combines features per slide to generate structured datasets.  
- **Output**: Saves processed features for downstream statistical or machine learning analysis.
