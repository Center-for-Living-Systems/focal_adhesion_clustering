# Focal Adhesion Clustering

This repository implements a full pipeline for **segmenting and clustering focal adhesions (FAs)** from 2D multi-channel fluorescence microscopy images. It uses **image filtering, local thresholding, and morphological analysis** to extract high-confidence FA objects, followed by **unsupervised clustering** based on both **morphological** and **spatial features**.


## Project Overview

Focal adhesions are highly dynamic and structurally diverse. This project aims to uncover **distinct FA subtypes** using **unsupervised learning** on features extracted from individual FA objects. The core steps include:

* Segmentation using **vesselness**, **dot filters**, and **adaptive thresholding**
* Feature extraction from segmented FA regions
* Clustering based on shape, intensity, and spatial context within the cell

## Data Description
* **Input**: Multi-channel 2D fluorescence microscopy images  
  * **Channel 1~3: Focal adhesion marker (e.g., paxillin, vinculin, zyxin, etc.)
  * **Channel last: Actin (phalloidin) used for defining cell boundaries

## Pipeline Overview  
### 1. **FA Segmentation**  
* Apply **vesselness and dot filtering** to enhance linear and blob-like FA structures
* Combine filters, apply **local thresholding**
* Use morphological operations to clean and isolate FA objects

### 2. **Cell Masking and Spatial Context**
* Segment cell outlines from the actin channel
* Compute **cell boundary contours** and **local normal vectors**

### 3. **Feature Extraction**
For each FA object:
* **Morphological Features**:
  * Area, eccentricity, major-axis-length
  * Intensity-based metrics (mean)
* **Spatial Features**:
  * Distance to nearest point on the cell boundary
  * Angle between FA orientation and boundary normal at closest point

### 4. **Unsupervised Clustering**  
* Concatenate features into a structured feature matrix
* Normalize features and reduce dimensionality (e.g., PCA, UMAP)
* Cluster FAs using KMeans, GMM, or HDBSCAN
* Visualize clusters in 2D embeddings and FA morphology montages

## Goals  
* **Discover and characterize focal adhesion phenotypes** using unsupervised methods
* Understand structural diversity without relying on manual labels
* Use both **shape and spatial context** to define biologically meaningful FA subtypes

## Requirements  
* Python ≥ 3.9
* `numpy`, `pandas`, `scikit-image`, `scikit-learn`, `aics-segmentation`
* `umap-learn` (for clustering and dimensionality reduction)
* `matplotlib`, `seaborn` (for visualization)
* `tifffile`, `czifile` (for image I/O)

## Outputs  
* Segmented focal adhesion masks
* Per-object feature table (`.csv`)
* Cluster labels for each FA object
* Dimensionality-reduced visualizations (UMAP)
* Cluster montages showing FA structure variability
* Summary statistics per image or condition (for downstream use)
