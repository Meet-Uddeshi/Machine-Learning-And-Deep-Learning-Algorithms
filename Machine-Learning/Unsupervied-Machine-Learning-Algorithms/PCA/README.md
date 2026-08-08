# Principal Component Analysis (PCA) Module

An end-to-end Object-Oriented machine learning pipeline implementing Principal Component Analysis (PCA) for linear dimensionality reduction and feature compression.

---

## Architecture and Directory Structure

The project follows a clean Service-Oriented Architecture (SOA) where data loading, preprocessing, numerical standardization, covariance matrix computation, eigendecomposition, and visualization routines are isolated into dedicated service classes inside `src/`. Thin orchestration scripts manage pipeline execution, while immutable dataclasses handle configuration settings.

```text
PCA/
│
├── README.md                          # Comprehensive module guide & documentation
│
├── data/                              # Input dataset folder
│   └── pca.csv                        # Vehicle Silhouette continuous shape metrics dataset
│
├── output/                            # Generated evaluation outputs and figures
│   ├── scree_plot.png                 # Individual & cumulative variance scree plot
│   ├── pca_2d_projection.png          # 2D PC1 vs PC2 projection scatter plot
│   └── pca_analysis_report.md         # Analytical summary report
│
└── src/                               # Python source codebase
    ├── config.py                      # Immutable configuration dataclasses
    ├── logger.py                      # Centralized stdout logger factory
    ├── data_loader.py                 # Data loading, imputation, & scaling service
    ├── pca_service.py                 # Core PCA computation & visualization service
    └── main.py                        # Main pipeline orchestration entry point
```

---

## Mathematical Foundations of PCA

Principal Component Analysis (PCA) transforms a set of correlated continuous features $X \in \mathbb{R}^{n \times p}$ into an orthogonal set of linearly uncorrelated variables called Principal Components $Z \in \mathbb{R}^{n \times k}$ ($k \le p$), maximizing variance captured along orthogonal directions.

### Step 1: Feature Standardization
Data is centered and scaled to zero mean and unit variance:
$$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$

### Step 2: Covariance Matrix Computation
The $p \times p$ sample covariance matrix $\Sigma$ is computed:
$$\Sigma = \frac{1}{n - 1} Z^T Z$$

### Step 3: Eigendecomposition
Solving the characteristic equation for eigenvalues $\lambda_i$ and eigenvectors $v_i$:
$$\Sigma v_i = \lambda_i v_i$$

Eigenvectors $v_i$ define the principal directions (loadings), and eigenvalues $\lambda_i$ represent the variance explained by component $i$.

### Step 4: Explained Variance Ratio & Selection
The proportion of total variance explained by the $i$-th principal component is:
$$\text{Ratio}_i = \frac{\lambda_i}{\sum_{j=1}^{p} \lambda_j}$$

Components are selected until the cumulative explained variance ratio meets the target threshold (e.g., 95%):
$$\text{Cumulative Variance}(k) = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{j=1}^{p} \lambda_j} \ge 0.95$$

### Step 5: Low-Dimensional Projection & Reconstruction
The dataset is projected into $k$-dimensional space using projection matrix $V_k = [v_1, v_2, \dots, v_k]$:
$$X_{\text{pca}} = Z V_k$$

Reconstructed approximation in original feature space:
$$\hat{Z} = X_{\text{pca}} V_k^T$$

Reconstruction Mean Squared Error (MSE):
$$\text{MSE} = \frac{1}{n \cdot p} \sum_{i=1}^{n} \sum_{j=1}^{p} (z_{ij} - \hat{z}_{ij})^2$$

---

## Dataset Description (`pca.csv`)

The dataset contains 18 continuous numerical shape measurements of 3D vehicle silhouettes (buses, cars, vans):
- **Features**: `compactness`, `circularity`, `distance_circularity`, `radius_ratio`, `pr.axis_aspect_ratio`, `max.length_aspect_ratio`, `scatter_ratio`, `elongatedness`, `pr.axis_rectangularity`, `max.length_rectangularity`, `scaled_variance`, `scaled_variance.1`, `scaled_radius_of_gyration`, `scaled_radius_of_gyration.1`, `skewness_about`, `skewness_about.1`, `skewness_about.2`, `hollows_ratio`.
- **Target**: `class` (`bus`, `car`, `van`, etc.).

---

## Execution Instructions

Execute the main orchestration script from your terminal:

```bash
python Machine-Learning/Unsupervied-Machine-Learning-Algorithms/PCA/src/main.py
```

All generated visualization figures (`scree_plot.png`, `pca_2d_projection.png`) and analytical report (`pca_analysis_report.md`) are stored in the `output/` directory.
