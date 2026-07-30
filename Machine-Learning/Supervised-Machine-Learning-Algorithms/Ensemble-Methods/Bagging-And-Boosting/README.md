# Bagging and Boosting Ensemble Learning Module

An end-to-end Object-Oriented machine learning pipeline implementing Bagging (Bootstrap Aggregating), AdaBoost (Adaptive Boosting), and Gradient Boosting ensemble classifiers on tabular medical dataset (`heart.csv`).

---

## Architecture and Directory Structure

The project follows a clean Service-Oriented Architecture (SOA) where data loading, preprocessing, model training, and evaluation routines are isolated into dedicated service classes inside `src/`. Thin orchestration scripts manage pipeline execution, while immutable dataclasses handle configuration settings.

```text
Bagging-And-Boosting/
│
├── README.md                          # Comprehensive module guide & documentation
│
├── data/                              # Input dataset folder
│   └── heart.csv                      # Heart Disease classification dataset
│
├── output/                            # Generated evaluation outputs and figures
│   ├── confusion_matrix_bagging.png
│   ├── confusion_matrix_adaboost.png
│   ├── confusion_matrix_gradientboosting.png
│   ├── ensemble_comparison_bar_chart.png
│   └── bagging_boosting_analysis_report.md
│
└── src/                               # Python source codebase
    ├── config.py                      # Immutable configuration dataclasses
    ├── logger.py                      # Centralized stdout logger factory
    ├── data_loader.py                 # Data loading, scaling, & splitting service
    ├── ensemble_classifier.py         # Ensemble training & evaluation service
    └── main.py                        # Main pipeline orchestration entry point
```

---

## Key Concepts and Algorithms Implemented

### 1. Bagging (Bootstrap Aggregating)
- **Concept**: Reduces variance by training multiple base decision trees in parallel on bootstrap random sub-samples of the training dataset with replacement.
- **Formulation**: For an ensemble of $M$ decision trees $f_m(x)$, the aggregated prediction is:
  $$\hat{f}_{\text{bag}}(x) = \frac{1}{M} \sum_{m=1}^{M} f_m(x)$$

### 2. AdaBoost (Adaptive Boosting)
- **Concept**: Iteratively fits weak base learners in sequence. After each iteration, weights of misclassified observations are increased so subsequent learners focus on difficult samples.
- **Formulation**: Final strong classifier output:
  $$H(x) = \text{sign}\left( \sum_{m=1}^{M} \alpha_m h_m(x) \right)$$
  where $\alpha_m = \frac{1}{2} \ln\left( \frac{1 - e_m}{e_m} \right)$ represents the estimator weight based on error rate $e_m$.

### 3. Gradient Boosting
- **Concept**: Sequentially fits decision trees where each new tree models the negative gradient (pseudo-residuals) of the loss function $L(y, f(x))$ from the previous iteration.
- **Formulation**: Model update at step $m$:
  $$f_m(x) = f_{m-1}(x) + \eta \cdot h_m(x)$$
  where $\eta$ is the shrinkage learning rate ($0 < \eta \le 1$).

---

## Dataset Description (`heart.csv`)

The model evaluates clinical attributes to predict binary heart disease presence (`target` = 0 or 1):
- **Features**: `age`, `sex`, `cp` (chest pain type), `trestbps` (resting blood pressure), `chol` (cholesterol), `fbs` (fasting blood sugar), `restecg`, `thalach` (max heart rate), `exang`, `oldpeak`, `slope`, `ca`, `thal`.
- **Target**: `target` (0 = no disease, 1 = disease present).

---

## Evaluation Metrics & Outputs

The pipeline evaluates and records the following metrics for all three ensemble algorithms:
- **Accuracy**: Overall proportion of correct predictions.
- **Precision**: Proportion of true positive disease predictions.
- **Recall (Sensitivity)**: Proportion of actual positive disease cases correctly identified.
- **F1 Score**: Harmonic mean of Precision and Recall.
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve.

Figures generated in `output/`:
- `confusion_matrix_bagging.png`
- `confusion_matrix_adaboost.png`
- `confusion_matrix_gradientboosting.png`
- `ensemble_comparison_bar_chart.png`

---

## Execution Instructions

Execute the main orchestration script from your terminal:

```bash
python Machine-Learning/Supervised-Machine-Learning-Algorithms/Ensemble-Methods/Bagging-And-Boosting/src/main.py
```
