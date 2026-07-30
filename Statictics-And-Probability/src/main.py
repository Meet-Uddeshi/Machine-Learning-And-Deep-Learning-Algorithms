# ============================================================================
# Main Entry Point -- GPU Statistics and Probability Processing Pipeline
# ============================================================================
# Orchestration script that wires together configuration, logging, data loading
# for `gpu_database.csv`, and executes the complete suite of analytical
# algorithms from Probability.png and Statistics.png.
# Stores all figures, log files, and markdown report in the output/ directory.
# ============================================================================

import os
import sys
import time

# Ensure src directory is in Python path for flexible invocation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from config import PipelineConfig
from data_loader import DataLoaderService
from descriptive_stats import DescriptiveStatisticsService
from inferential_stats import InferentialStatisticsService
from limit_theorems import LimitTheoremsService
from multivariate_stats import MultivariateStatsService
from non_parametric import NonParametricService
from probability_distributions import DistributionsService
from probability_theory import ProbabilityService
from regression_and_correlation import RegressionCorrelationService
from sampling_and_experiments import SamplingAndDesignService
from time_series import TimeSeriesService


def main() -> None:
    """Orchestrate the end-to-end GPU Statistics & Probability analytical pipeline.

    Execution Pipeline:
        1. Ingest configuration parameters.
        2. Set up logging to stdout and output log file.
        3. Load and preprocess `gpu_database.csv`.
        4. Execute Descriptive Statistics & save `descriptive_summaries.png`.
        5. Execute Probability Theory calculations (empirical GPU probabilities, Bayes, bounds).
        6. Evaluate Probability Distributions & save `probability_distributions.png`.
        7. Run Monte Carlo simulations for LLN and CLT & save `clt_and_lln_demonstration.png`.
        8. Perform Sampling strategies (Simple, Systematic, Stratified, Cluster).
        9. Run Inferential Statistics (Z/T Intervals, T-tests, Chi-square, ANOVA).
        10. Fit Regression & Correlation & save `regression_analysis_plots.png`.
        11. Run Time Series Decomposition & save `time_series_decomposition_additive.png`.
        12. Run Non-Parametric Rank Tests (Mann-Whitney U, Kruskal-Wallis, Spearman).
        13. Run Multivariate Statistics (Multiple Regression & Covariance Matrix).
        14. Compile and write comprehensive output report (`statistics_and_probability_report.md`).
    """
    pipeline_start = time.perf_counter()

    # Step 1: Configuration Ingestion
    config = PipelineConfig()

    # Step 2: Logger Setup
    from logger import LoggerFactory
    logger = LoggerFactory.create(
        name="GPU-Statistics-Pipeline",
        logging_config=config.logging,
        path_config=config.paths,
    )

    logger.info("=" * 75)
    logger.info("GPU DATABASE STATISTICS AND PROBABILITY PIPELINE")
    logger.info("=" * 75)
    logger.info("Dataset File   : %s", config.paths.dataset_file)
    logger.info("Output Folder  : %s", config.paths.output_dir)

    try:
        # Step 3: Load and Preprocess GPU Dataset
        data_service = DataLoaderService(
            path_config=config.paths,
            data_config=config.data,
            logger=logger,
        )
        df = data_service.load_and_prepare_data()
        ts_gflops = data_service.get_time_series_aggregated(df)

        # Step 4: Descriptive Statistics Service
        desc_service = DescriptiveStatisticsService(
            path_config=config.paths, logger=logger
        )
        desc_metrics = desc_service.compute_descriptive_stats(
            df["die_size_mm2"], name="die_size_mm2"
        )
        desc_plot = desc_service.generate_visual_summaries(
            df, continuous_col="die_size_mm2", categorical_col="manufacturer_clean"
        )

        # Step 5: Probability Theory Service
        prob_service = ProbabilityService(logger=logger)
        empirical_probs = prob_service.compute_empirical_gpu_probabilities(df)
        p_classical = prob_service.compute_classical_probability(
            favorable_outcomes=6, total_outcomes=36
        )
        p_union = prob_service.addition_rule(
            p_a=empirical_probs["P(TDP > 150W)"],
            p_b=empirical_probs["P(Nvidia)"],
            p_a_and_b=empirical_probs["P(TDP > 150W and Nvidia)"],
        )
        bayes_res = prob_service.bayes_theorem(
            prior_p_b=empirical_probs["P(Nvidia)"],
            sensitivity_p_a_given_b=empirical_probs["P(TDP > 150W | Nvidia)"],
            false_positive_p_a_given_b_c=0.15,
        )
        ineq_res = prob_service.evaluate_inequalities(
            mean=float(df["tdp_watts"].mean()),
            variance=float(df["tdp_watts"].var()),
            a=200.0,
            k=2.0,
            probabilities=[0.25, 0.30, 0.15],
        )

        # Step 6: Probability Distributions Service
        dist_service = DistributionsService(
            analysis_config=config.analysis,
            path_config=config.paths,
            logger=logger,
        )
        binom_res = dist_service.evaluate_binomial(k=10)
        poisson_res = dist_service.evaluate_poisson(k=4)
        normal_res = dist_service.evaluate_normal(x=float(df["core_clock_mhz"].mean()))
        dist_plot = dist_service.generate_distribution_plots()

        # Step 7: Limit Theorems Service
        limit_service = LimitTheoremsService(
            path_config=config.paths, logger=logger
        )
        max_trans_gpu = df["transistors_million"].max()
        z_score_val = limit_service.compute_z_score(
            x=float(max_trans_gpu),
            mean=float(df["transistors_million"].mean()),
            std_dev=float(df["transistors_million"].std()),
        )
        clt_plot = limit_service.simulate_lln_and_clt()

        # Step 8: Sampling and Experimental Design Service
        sampling_service = SamplingAndDesignService(logger=logger)
        simple_sample = sampling_service.simple_random_sample(df, n_samples=50)
        stratified_sample = sampling_service.stratified_sample(
            df, strata_column="manufacturer_clean", frac=0.1
        )
        exp_designs = SamplingAndDesignService.get_experimental_designs_summary()

        # Step 9: Inferential Statistics Service
        infer_service = InferentialStatisticsService(
            data_config=config.data, logger=logger
        )
        ci_res = infer_service.confidence_interval_mean(df["die_size_mm2"].values)
        t_test_res = infer_service.one_sample_t_test(
            df["tdp_watts"].values, pop_mean_h0=100.0
        )

        nvidia_gflops = df[df["manufacturer_clean"] == "Nvidia"]["processing_power_gflops"].values
        amd_gflops = df[df["manufacturer_clean"] == "AMD"]["processing_power_gflops"].values
        two_sample_t_res = infer_service.two_sample_t_test(
            nvidia_gflops, amd_gflops, paired=False
        )

        contingency_tab = pd.crosstab(df["manufacturer_clean"], df["bus_interface"])
        chi2_res = infer_service.chi_square_independence_test(contingency_tab)

        mfg_groups = [
            group["processing_power_gflops"].values
            for _, group in df.groupby("manufacturer_clean")
            if len(group) >= 5
        ]
        anova_res = infer_service.one_way_anova(*mfg_groups)

        # Step 10: Regression and Correlation Service
        reg_service = RegressionCorrelationService(
            path_config=config.paths, logger=logger
        )
        r_val = reg_service.compute_pearson_correlation(
            df["transistors_million"].values, df["processing_power_gflops"].values
        )
        ols_res = reg_service.fit_simple_linear_regression(
            df["transistors_million"].values, df["processing_power_gflops"].values
        )

        # Step 11: Time Series Service
        ts_service = TimeSeriesService(
            path_config=config.paths, logger=logger
        )
        ts_decomp = ts_service.decompose_time_series(
            ts_gflops, model="additive", period=3
        )

        # Step 12: Non-Parametric Service
        nonpar_service = NonParametricService(
            data_config=config.data, logger=logger
        )
        nvidia_tdp = df[df["manufacturer_clean"] == "Nvidia"]["tdp_watts"].values
        amd_tdp = df[df["manufacturer_clean"] == "AMD"]["tdp_watts"].values
        mwu_res = nonpar_service.mann_whitney_u_test(nvidia_tdp, amd_tdp)
        kw_res = nonpar_service.kruskal_wallis_test(*mfg_groups)
        spearman_res = nonpar_service.spearman_rank_correlation(
            df["die_size_mm2"].values, df["tdp_watts"].values
        )

        # Step 13: Multivariate Statistics Service
        multi_service = MultivariateStatsService(
            data_config=config.data, logger=logger
        )
        x_multi = df[
            [
                "transistors_million",
                "die_size_mm2",
                "core_clock_mhz",
                "tdp_watts",
            ]
        ].values
        y_multi = df["processing_power_gflops"].values
        multi_reg_res = multi_service.fit_multiple_linear_regression(
            x_multi,
            y_multi,
            feature_names=[
                "transistors_million",
                "die_size_mm2",
                "core_clock_mhz",
                "tdp_watts",
            ],
        )
        cov_mat = multi_service.compute_covariance_matrix(
            df[
                [
                    "transistors_million",
                    "die_size_mm2",
                    "core_clock_mhz",
                    "processing_power_gflops",
                    "tdp_watts",
                ]
            ]
        )

        # Step 14: Technical Summary Markdown Report Generation
        write_summary_report(
            output_dir=config.paths.output_dir,
            df_shape=df.shape,
            desc_metrics=desc_metrics,
            empirical_probs=empirical_probs,
            bayes_res=bayes_res,
            ineq_res=ineq_res,
            ci_res=ci_res,
            t_test_res=t_test_res,
            two_sample_t_res=two_sample_t_res,
            chi2_res=chi2_res,
            anova_res=anova_res,
            r_val=r_val,
            ols_res=ols_res,
            mwu_res=mwu_res,
            spearman_res=spearman_res,
            multi_reg_res=multi_reg_res,
        )

    except Exception as exc:
        logger.exception("Pipeline execution encountered an unexpected error: %s", exc)
        sys.exit(1)

    elapsed = time.perf_counter() - pipeline_start
    logger.info("=" * 75)
    logger.info("GPU Pipeline executed successfully in %.4f seconds.", elapsed)
    logger.info("=" * 75)


def write_summary_report(
    output_dir: str,
    df_shape: tuple,
    desc_metrics: dict,
    empirical_probs: dict,
    bayes_res: dict,
    ineq_res: dict,
    ci_res: dict,
    t_test_res: dict,
    two_sample_t_res: dict,
    chi2_res: dict,
    anova_res: dict,
    r_val: float,
    ols_res: dict,
    mwu_res: dict,
    spearman_res: dict,
    multi_reg_res: dict,
) -> None:
    """Write comprehensive analytical GPU dataset report to markdown.

    Args:
        output_dir:       Destination output folder.
        df_shape:         Tuple of (rows, cols).
        desc_metrics:     Descriptive statistics dictionary.
        empirical_probs:  Empirical GPU probability dictionary.
        bayes_res:        Bayes theorem dictionary.
        ineq_res:         Inequalities dictionary.
        ci_res:           Confidence interval dictionary.
        t_test_res:       One-sample T-test dictionary.
        two_sample_t_res: Two-sample T-test dictionary.
        chi2_res:         Chi-square test dictionary.
        anova_res:        ANOVA test dictionary.
        r_val:            Pearson correlation r.
        ols_res:          OLS regression dictionary.
        mwu_res:          Mann-Whitney U dictionary.
        spearman_res:     Spearman rank correlation dictionary.
        multi_reg_res:    Multiple regression dictionary.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "statistics_and_probability_report.md")

    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("# GPU Database Statistics & Probability Analysis Report\n\n")

        fh.write("## 1. Executive Summary & Dataset Overview\n")
        fh.write(f"Analyzed real-world GPU database containing **{df_shape[0]}** GPU models across major hardware manufacturers ")
        fh.write("(Nvidia, AMD/ATI, Intel) spanning 1995 to 2023.\n\n")

        fh.write("## 2. Descriptive Statistics & Visual Summaries\n")
        fh.write("Measures of central tendency and dispersion for GPU `die_size_mm2`:\n\n")
        fh.write("| Metric | Value | Metric | Value |\n")
        fh.write("|--------|-------|--------|-------|\n")
        fh.write(f"| **Mean** | {desc_metrics['Mean']:.2f} mm² | **Variance ($s^2$)** | {desc_metrics['Variance']:.2f} |\n")
        fh.write(f"| **Median** | {desc_metrics['Median']:.2f} mm² | **Std Dev ($s$)** | {desc_metrics['Std_Dev']:.2f} mm² |\n")
        fh.write(f"| **Mode** | {desc_metrics['Mode']:.2f} mm² | **Range** | {desc_metrics['Range']:.2f} mm² |\n")
        fh.write(f"| **Q1 (25th)** | {desc_metrics['Q1']:.2f} mm² | **Q3 (75th)** | {desc_metrics['Q3']:.2f} mm² |\n")
        fh.write(f"| **IQR** | {desc_metrics['IQR']:.2f} mm² | **Total Count** | {desc_metrics['Count']} |\n\n")

        fh.write("## 3. Probability Theory & Empirical GPU Analysis\n")
        fh.write("- **Empirical Event Probabilities**:\n")
        fh.write(f"  - $P(\\text{{TDP}} > 150\\text{{W}})$: {empirical_probs['P(TDP > 150W)']:.4f}\n")
        fh.write(f"  - $P(\\text{{Nvidia}})$: {empirical_probs['P(Nvidia)']:.4f}\n")
        fh.write(f"  - $P(\\text{{TDP}} > 150\\text{{W}} \\mid \\text{{Nvidia}})$: **{empirical_probs['P(TDP > 150W | Nvidia)'] * 100:.2f}%**\n\n")

        fh.write("- **Bayes' Theorem Evaluation**:\n")
        fh.write(f"  - Posterior Probability $P(\\text{{Nvidia}} \\mid \\text{{High Power}}): {bayes_res['Posterior_P(B|A)'] * 100:.2f}\\%$\n\n")

        fh.write("- **Probability Inequalities Upper Bounds (GPU Power TDP)**:\n")
        fh.write(f"  - Markov's Bound $P(X \\ge 200\\text{{W}}) \\le {ineq_res['Markov_Bound']:.4f}$\n")
        fh.write(f"  - Chebyshev's Bound $P(|X - \\mu| \\ge 2\\sigma) \\le {ineq_res['Chebyshev_Bound']:.4f}$\n")
        fh.write(f"  - Union Bound $P(\\bigcup A_i) \\le {ineq_res['Union_Bound']:.4f}$\n\n")

        fh.write("## 4. Inferential Statistics & Hypothesis Testing\n")
        fh.write(f"- **95% Confidence Interval for Die Size Mean**: [{ci_res['Lower_CI']:.2f}, {ci_res['Upper_CI']:.2f}] mm² (Margin of Error: {ci_res['Margin_of_Error']:.2f})\n")
        fh.write(f"- **One-Sample T-Test ($H_0: \\text{{Mean TDP}} = 100\\text{{W}}$)**: $t = {t_test_res['Test_Statistic']:.4f}, p = {t_test_res['P_Value']:.4e} \\rightarrow$ **{t_test_res['Decision']}**\n")
        fh.write(f"- **Two-Sample T-Test (Nvidia vs AMD GFLOPS)**: $t = {two_sample_t_res['Test_Statistic']:.4f}, p = {two_sample_t_res['P_Value']:.4e} \\rightarrow$ **{two_sample_t_res['Decision']}**\n")
        fh.write(f"- **Chi-Square Independence Test (Manufacturer vs Bus Interface)**: $\\chi^2 = {chi2_res['Chi2_Statistic']:.4f}, p = {chi2_res['P_Value']:.4e} \\rightarrow$ **{chi2_res['Decision']}**\n")
        fh.write(f"- **One-Way ANOVA (GFLOPS across Manufacturers)**: $F = {anova_res['F_Statistic']:.4f}, p = {anova_res['P_Value']:.4e} \\rightarrow$ **{anova_res['Decision']}**\n\n")

        fh.write("## 5. Correlation & Linear Regression Diagnostics\n")
        fh.write(f"- **Pearson Correlation ($r$) (Transistors vs GFLOPS)**: **{r_val:.4f}**\n")
        fh.write(f"- **Spearman Rank Correlation ($\\rho$) (Die Size vs TDP)**: **{spearman_res['Spearman_Rho']:.4f}**\n")
        fh.write(f"- **Simple OLS Regression Equation**: $\\hat{{\\text{{GFLOPS}}}} = {ols_res['Intercept_Beta0']:.4f} + {ols_res['Slope_Beta1']:.4f} \\cdot \\text{{Transistors(M)}}$\n")
        fh.write(f"- **Simple Regression $R^2$**: **{ols_res['R_Squared']:.4f}**\n\n")

        fh.write("## 6. Non-Parametric Tests\n")
        fh.write(f"- **Mann-Whitney U Test (Nvidia vs AMD TDP)**: $U = {mwu_res['U_Statistic']:.4f}, p = {mwu_res['P_Value']:.4e} \\rightarrow$ **{mwu_res['Decision']}**\n\n")

        fh.write("## 7. Multiple Linear Regression Model\n")
        fh.write(f"- **Multiple Regression $R^2$**: **{multi_reg_res['R_Squared']:.4f}**\n")
        fh.write(f"- **Intercept**: {multi_reg_res['Intercept']:.4f}\n")
        fh.write("- **Feature Coefficients**:\n")
        for k, v in multi_reg_res["Coefficients"].items():
            fh.write(f"  - `{k}`: {v:+.4f}\n")


if __name__ == "__main__":
    main()
