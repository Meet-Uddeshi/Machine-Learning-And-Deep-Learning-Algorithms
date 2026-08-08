# Unsupervised Machine Learning - Principal Component Analysis (PCA)

> Unsupervised Machine Learning | Dimensionality Reduction & Feature Decomposition Algorithm

---

## Table of Contents

1. [What is PCA?](#1-what-is-pca)
2. [Theoretical Explanation](#2-theoretical-explanation)
3. [Mathematical Operations](#3-mathematical-operations)
4. [Real-World Example](#4-real-world-example)
5. [Worked PCA Sum (Step-by-Step)](#5-worked-pca-sum-step-by-step)
6. [Program Flowchart](#6-program-flowchart)
7. [Module Responsibility Map](#7-module-responsibility-map)
8. [Configuration](#8-configuration)

---

## 1. What is PCA?

Principal Component Analysis (PCA) is an **unsupervised linear dimensionality reduction technique**. It transforms a high-dimensional space of correlated features into a lower-dimensional subspace of linearly uncorrelated variables called **Principal Components (PCs)**.

PCA performs an orthogonal linear transformation such that:
- The first principal component ($PC_1$) accounts for the largest possible variance in the data.
- Each succeeding component ($PC_k$) has the highest possible variance under the constraint that it is orthogonal (perpendicular) to all preceding components.

### Key Characteristics

| Property           | Description                                                          |
|--------------------|----------------------------------------------------------------------|
| Task type          | Unsupervised Learning                                                |
| Transformation     | Linear orthogonal rotation of coordinate axes                       |
| Primary Objectives | Variance maximization, feature decorrelation, noise reduction        |
| Features output    | Ordered orthogonal components ($PC_1, PC_2, \dots, PC_K$)            |
| Key Metrics        | Eigenvalues, Explained Variance Ratio, Cumulative Variance Ratio     |

---

## 2. Theoretical Explanation

### How PCA Works

PCA re-orients the dataset along axes of maximum variance. Given a dataset with $D$ features, PCA computes $D$ orthogonal eigenvectors.

```
High-Dimensional Correlated Features ---> Standardise Features ---> Compute Covariance Matrix
                                                                           |
                                                                           v
         Reconstructed Subspace / Plots <--- Project Data Vectors <--- Eigendecomposition / SVD
```

1. **Mean-Centering & Scaling**: Subtract column means and divide by feature standard deviations.
2. **Covariance Matrix Construction**: Capture pairwise feature correlations.
3. **Eigen-Decomposition / SVD**: Extract eigenvalues ($\lambda$) and eigenvectors ($v$). Eigenvectors define the direction of the principal axes; eigenvalues quantify the variance along each axis.
4. **Subspace Projection**: Multiply the original feature matrix by the top $K$ eigenvectors to obtain low-dimensional embeddings.

### Why Feature Standardisation is Mandatory for PCA
PCA seeks directions that maximize variance. If one feature is measured in millimeters (range $0-1000$) and another in meters (range $0-1$), the first feature will artificially dominate the variance calculation simply due to its scale. 

Applying **Z-score standardisation** ($z = \frac{x - \mu}{\sigma}$) gives every feature a mean of $0$ and a variance of $1$, ensuring equal weighting in direction optimization.

---

## 3. Mathematical Operations

### 1. Feature Standardisation

Given data matrix $X \in \mathbb{R}^{N \times D}$, standardise each feature column $j$:

$$Z_{i,j} = \frac{X_{i,j} - \mu_j}{\sigma_j}$$

where $\mu_j = \frac{1}{N} \sum_{i=1}^N X_{i,j}$ and $\sigma_j = \sqrt{\frac{1}{N-1} \sum_{i=1}^N (X_{i,j} - \mu_j)^2}$.

---

### 2. Sample Covariance Matrix

Compute the $D \times D$ covariance matrix $\Sigma$:

$$\Sigma = \frac{1}{N-1} Z^T Z$$

where element $\Sigma_{j,k} = \text{cov}(Z_{\cdot, j}, Z_{\cdot, k})$.

---

### 3. Eigendecomposition

Solve the characteristic equation for eigenvalues $\lambda_k$ and eigenvectors $v_k$:

$$\Sigma v_k = \lambda_k v_k$$

Alternatively, using **Singular Value Decomposition (SVD)** directly on $Z$:

$$Z = U S V^T$$

where:
- $V \in \mathbb{R}^{D \times D}$ contains the principal directions (eigenvectors).
- $S \in \mathbb{R}^{N \times D}$ contains singular values $s_k = \sqrt{(N-1) \lambda_k}$.

---

### 4. Explained Variance Ratios

The proportion of total variance explained by the $k$-th principal component is:

$$\text{Explained Variance Ratio}_k = \frac{\lambda_k}{\sum_{j=1}^D \lambda_j}$$

The cumulative explained variance for $K$ components is:

$$\text{Cumulative Variance}_K = \frac{\sum_{k=1}^K \lambda_k}{\sum_{j=1}^D \lambda_j}$$

---

### 5. Subspace Projection

Select the top $K$ eigenvectors to form matrix $W_K \in \mathbb{R}^{D \times K}$. Transform the standardized dataset $Z$ into lower dimensions $Z_{\text{pca}} \in \mathbb{R}^{N \times K}$:

$$Z_{\text{pca}} = Z W_K$$

---

## 4. Real-World Example

### Vehicle Silhouettes Dimensionality Reduction
The pipeline processes 18 geometrical features extracted from 2D silhouette images of vehicles (`pca.csv`).

- **Dataset**: `pca.csv` (Vehicle Silhouettes Dataset)
- **Target Variable**: `class` (`bus`, `car`, `van` - used solely for color-coded 2D scatter plots)
- **Features Used (18 Continuous Attributes)**:
  `compactness`, `circularity`, `distance_circularity`, `radius_ratio`, `pr.axis_aspect_ratio`, `max.length_aspect_ratio`, `scatter_ratio`, `elongatedness`, `pr.axis_rectangularity`, `max.length_rectangularity`, `scaled_variance`, `scaled_variance.1`, `scaled_radius_of_gyration`, `scaled_radius_of_gyration.1`, `skewness_about`, `skewness_about.1`, `skewness_about.2`, `hollows_ratio`.

---

## 5. Worked PCA Sum (Step-by-Step)

Let us trace PCA on a toy 2D dataset with $N=3$ observations and $D=2$ features:
- $P_1 = (2, 4)$
- $P_2 = (3, 5)$
- $P_3 = (4, 6)$

### Step 1: Compute Feature Means
$$\mu_1 = \frac{2+3+4}{3} = 3, \quad \mu_2 = \frac{4+5+6}{3} = 5$$

---

### Step 2: Mean-Center the Dataset ($Z$)
Subtract column means:
- $Z_1 = (2-3, 4-5) = (-1, -1)$
- $Z_2 = (3-3, 5-5) = (0, 0)$
- $Z_3 = (4-3, 6-5) = (1, 1)$

$$Z = \begin{bmatrix} -1 & -1 \\ 0 & 0 \\ 1 & 1 \end{bmatrix}$$

---

### Step 3: Compute Covariance Matrix ($\Sigma$)
$$\Sigma = \frac{1}{3-1} Z^T Z = \frac{1}{2} \begin{bmatrix} -1 & 0 & 1 \\ -1 & 0 & 1 \end{bmatrix} \begin{bmatrix} -1 & -1 \\ 0 & 0 \\ 1 & 1 \end{bmatrix} = \frac{1}{2} \begin{bmatrix} 2 & 2 \\ 2 & 2 \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}$$

---

### Step 4: Compute Eigenvalues & Eigenvectors
Solve $\det(\Sigma - \lambda I) = 0$:

$$\det \begin{bmatrix} 1-\lambda & 1 \\ 1 & 1-\lambda \end{bmatrix} = (1-\lambda)^2 - 1 = \lambda^2 - 2\lambda = 0$$

Roots:
- $\lambda_1 = 2$
- $\lambda_2 = 0$

#### Eigenvector for $\lambda_1 = 2$:
$$\begin{bmatrix} 1-2 & 1 \\ 1 & 1-2 \end{bmatrix} \begin{bmatrix} v_{11} \\ v_{12} \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \implies -v_{11} + v_{12} = 0 \implies v_{11} = v_{12}$$

Normalized unit eigenvector:
$$v_1 = \begin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{bmatrix} \approx \begin{bmatrix} 0.7071 \\ 0.7071 \end{bmatrix}$$

Total Variance = $\lambda_1 + \lambda_2 = 2 + 0 = 2$.
Variance ratio for $PC_1 = \frac{2}{2} = 1.0$ (100% of variance explained by 1 component!).

---

### Step 5: Project Data Points onto $PC_1$
$$Z_{\text{pca1}} = Z v_1$$

- For $P_1$: $(-1)(0.7071) + (-1)(0.7071) = -1.4142$
- For $P_2$: $(0)(0.7071) + (0)(0.7071) = 0.0000$
- For $P_3$: $(1)(0.7071) + (1)(0.7071) = 1.4142$

The 2D dataset is perfectly projected onto a 1D line without any loss of information.

---

## 6. Program Flowchart

```
+-----------------------------------------------------+
|               START: main.py runs                   |
+-----------------------------------------------------+
                           |
                           v
+-----------------------------------------------------+
|  Step 1: Load PipelineConfig                        |
|  - PathConfig   (dataset and output paths)          |
|  - DataConfig   (class target, splits, scaling=True)|
|  - ModelConfig  (n_components=None, svd_solver)   |
|  - LoggingConfig (console-only logs)                |
+-----------------------------------------------------+
                           |
                           v
+-----------------------------------------------------+
|  Step 2: Initialize Console Logger                  |
|  - Output formatted to sys.stdout (No local files)  |
+-----------------------------------------------------+
                           |
                           v
+-----------------------------------------------------+
|  Step 3: DataLoaderService.load_and_prepare()       |
+-----------------------------------------------------+
                           |
           +---------------+-------------------+
           |                                   |
           v                                   v
+------------------+               +-------------------+
|  _load_csv()     |               |  File not found?  |
|  Read CSV from   |  -- Error --> |  Raise            |
|  data/ folder    |               |  FileNotFoundError|
+------------------+               +-------------------+
           |
           v
+------------------+
|  _validate_schema|
|  Check class     |
|  presence        |
+------------------+
           |
           v
+------------------+
|  _log_data_      |
|  summary()       |
|  Exploratory stats|
+------------------+
           |
           v
+--------------------------------------------------+
|  _preprocess()                                   |
|  - Separate target class ('class')               |
|  - Impute null values with column medians        |
+--------------------------------------------------+
           |
           v
+--------------------------------------------------+
|  _split() & standardise features                 |
|  - Split into 80% Train, 20% Test (stratified)   |
|  - Fit and transform StandardScaler on train set |
|  - Transform test set                            |
+--------------------------------------------------+
                           |
                           v
+-----------------------------------------------------+
|  Step 4: PCAService.train_and_evaluate()            |
+-----------------------------------------------------+
                           |
                           v
+--------------------------------------------------+
|  PCA Decomposition & Eigen-Analysis              |
|  - Instantiate PCA()                             |
|  - Fit and transform train set                   |
|  - Compute eigenvalues, singular values          |
|  - Compute explained variance ratio & cumulative |
|  - Extract loadings matrix                       |
+--------------------------------------------------+
                           |
                           v
+--------------------------------------------------+
|  Save Output Artifacts to output/                |
|  - Write pca_results.txt                         |
|  - Write pca_analysis.md report                  |
|  - Generate scree_plot.png                       |
|  - Generate pca_2d_projection.png                |
|  - Generate loadings_heatmap.png                 |
+--------------------------------------------------+
                           |
                           v
+-----------------------------------------------------+
|               END: Pipeline Complete               |
+-----------------------------------------------------+
```

---

## 7. Module Responsibility Map

```
main.py
  |
  +-- config.py       (PipelineConfig, PathConfig, DataConfig,
  |                    ModelConfig, LoggingConfig)
  |
  +-- logger.py       (LoggerFactory - console stream log setup)
  |
  +-- data_loader.py  (DataLoaderService - load, impute, scale features, splits)
  |
  +-- pca_service.py  (PCAService - fit, variance calculations, scree plot,
                       loadings heatmap, 2D projections, reporting)
```

---

## 8. Configuration

All parameters are configured in `src/config.py`.

| Parameter            | Location      | Default      | Description                                      |
|----------------------|---------------|--------------|--------------------------------------------------|
| `n_components`       | `ModelConfig` | `None`       | Number of components to keep (None keeps all)    |
| `svd_solver`         | `ModelConfig` | `'auto'`     | SVD solver strategy (`'auto'`, `'full'`, etc.)   |
| `whiten`             | `ModelConfig` | `False`      | Whitening transformation toggle                  |
| `scale_features`     | `DataConfig`  | `True`       | Apply Z-score StandardScaler prior to PCA        |
| `test_size`          | `DataConfig`  | `0.20`       | Partition percentage reserved for test evaluation|
| `target_column`      | `DataConfig`  | `'class'`    | Target class column for 2D plot color-coding     |