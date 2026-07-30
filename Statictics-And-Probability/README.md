# GPU Database Statistics and Probability Pipeline

An end-to-end Object-Oriented Statistics and Probability pipeline built on real-world GPU database hardware metrics (`gpu_database.csv`), covering all fundamental topics of Probability and Statistics illustrated in the visual course roadmaps (`Probability.png` and `Statistics.png`).

---

## Architecture and Project Structure

The project follows a clean Service-Oriented Architecture (SOA) where business logic, statistical algorithms, and data loading routines are isolated into dedicated service classes inside `src/`. Thin orchestration scripts manage pipeline execution, while immutable dataclasses handle configuration.

```text
Statictics-And-Probability/
│
├── Probability.png              # Visual roadmap for Probability concepts
├── Statistics.png               # Visual roadmap for Statistics concepts
├── README.md                    # Module documentation and usage guide
│
├── data/                        # Input datasets directory
│   └── gpu_database.csv         # Real-world GPU specs dataset (1945 GPU models)
│
├── output/                      # Pipeline outputs, logs, and figures
│   ├── descriptive_summaries.png
│   ├── probability_distributions.png
│   ├── clt_and_lln_demonstration.png
│   ├── regression_analysis_plots.png
│   ├── time_series_decomposition_additive.png
│   ├── statistics_and_probability.log
│   └── statistics_and_probability_report.md
│
└── src/                         # Python source codebase
    ├── config.py                # Immutable configuration dataclasses
    ├── logger.py                # Centralized logging factory
    ├── data_loader.py           # Preprocessing & GPU data loader service
    ├── descriptive_stats.py     # Central tendency and dispersion metrics
    ├── probability_theory.py    # Axioms, rules, Bayes' theorem, inequalities
    ├── probability_distributions.py # PMF/PDF for discrete & continuous models
    ├── limit_theorems.py        # LLN & CLT Monte Carlo simulations
    ├── sampling_and_experiments.py  # Sampling methods & experimental designs
    ├── inferential_stats.py     # Hypothesis testing & confidence intervals
    ├── regression_and_correlation.py# OLS regression & Pearson correlation
    ├── time_series.py           # Time series decomposition & smoothing
    ├── non_parametric.py        # Non-parametric rank tests
    ├── multivariate_stats.py    # Multiple regression & covariance matrix
    └── main.py                  # Main pipeline entry point
```

---

## Key Modules and Concepts Implemented

### 1. Data Loader & Preprocessing (`data_loader.py`)
- Ingests 1,945 GPU models across Nvidia, AMD/ATI, Intel, and others (1995–2023).
- Cleans missing numeric features (`transistors_million`, `die_size_mm2`, `core_clock_mhz`, `processing_power_gflops`, `tdp_watts`).
- Parses launch dates and aggregates time series metrics by launch year.

### 2. Descriptive Statistics (`descriptive_stats.py`)
- **Central Tendency**: Mean ($\bar{x} = \frac{\sum x_i}{n}$), Median, Mode.
- **Dispersion**: Range ($x_{\max} - x_{\min}$), Sample Variance ($s^2 = \frac{\sum (x_i - \bar{x})^2}{n - 1}$), Standard Deviation ($s = \sqrt{s^2}$), Interquartile Range ($\text{IQR} = Q_3 - Q_1$).
- **Visual Summaries**: Histograms of die size, box plots of TDP, manufacturer market share bar charts and pie charts.

### 3. Probability Theory (`probability_theory.py`)
- **Empirical GPU Probabilities**: $P(\text{TDP} > 150\text{W})$, $P(\text{Nvidia})$, $P(\text{TDP} > 150\text{W} \mid \text{Nvidia})$.
- **Axioms & Rules**: Complement, Addition, Conditional, Multiplication, and Independence testing.
- **Bayes' Theorem**: Posterior probability solver $P(B_j | A)$.
- **Probability Inequalities**: Markov's Inequality, Chebyshev's Inequality, Union Bound.

### 4. Probability Distributions (`probability_distributions.py`)
- **Discrete Distributions**: Bernoulli($p$), Binomial($n, p$), Geometric($p$), Poisson($\lambda$).
- **Continuous Distributions**: Uniform($a, b$), Normal($\mu, \sigma^2$), Exponential($\lambda$).
- Evaluates PMF/PDF, Expected Value $E[X]$, Variance $\text{Var}(X)$, and plots comparison curves.

### 5. Limit Theorems (`limit_theorems.py`)
- **Law of Large Numbers (LLN)**: Sample average convergence.
- **Central Limit Theorem (CLT)**: Sampling distribution of means bell curve convergence.
- **Z-Score Standardization**: $Z = \frac{X - \mu}{\sigma}$.

### 6. Sampling Methods & Experimental Design (`sampling_and_experiments.py`)
- Simple Random Sampling, Systematic Sampling, Stratified Sampling (by manufacturer), and Cluster Sampling.
- Experimental designs summary (CRD, RBD, Factorial).

### 7. Inferential Statistics (`inferential_stats.py`)
- **Confidence Intervals**: $Z$-interval ($\sigma$ known) and $T$-interval ($\sigma$ unknown).
- **Hypothesis Testing**: 1-Sample T-test, 2-Sample T-test (Nvidia vs AMD), Chi-Square Test of Independence, One-Way ANOVA.

### 8. Correlation & Regression (`regression_and_correlation.py`)
- **Pearson Correlation ($r$)**: Transistors vs Processing Power (GFLOPS).
- **Simple OLS Regression**: $\hat{y} = \beta_0 + \beta_1 x$ and $R^2$ evaluation.
- Diagnostic plots for regression line and residuals.

### 9. Time Series Analysis (`time_series.py`)
- Simple Moving Average (SMA) and Simple Exponential Smoothing (SES).
- Additive and Multiplicative seasonal decomposition of GPU performance evolution across years.

### 10. Non-Parametric Methods (`non_parametric.py`)
- Mann-Whitney U test, Wilcoxon Signed-Rank test, Kruskal-Wallis test, Spearman Rank Correlation ($\rho$).

### 11. Multivariate Statistics (`multivariate_stats.py`)
- Multiple OLS Linear Regression ($Y = X\beta + \varepsilon$) predicting GFLOPS from transistors, die size, core clock, and TDP.
- Multivariate Covariance Matrix.

---

## Execution Instructions

Execute the pipeline script from the terminal:

```bash
python Statictics-And-Probability/src/main.py
```

All generated figures (`*.png`), log files (`statistics_and_probability.log`), and executive markdown reports (`statistics_and_probability_report.md`) are stored in the `output/` directory.
