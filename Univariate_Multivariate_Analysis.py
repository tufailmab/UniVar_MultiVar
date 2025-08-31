"""
UniVar_MultiVar.py
-------------------

Author: Engr. Tufail Mabood, MSc Structural Engineering, UET Peshawar
Contact: https://wa.me/+923440907874
License: MIT License

Description:
This Python script performs comprehensive univariate and multivariate statistical analysis
on numeric datasets. It generates histograms, KDEs, boxplots, descriptive statistics,
pairplots for top predictors, and OLS regression outputs with coefficients, t-values,
and p-values. All results are saved as CSV, PNG, and text summaries, providing a 
reproducible workflow for feature evaluation, data exploration, and machine learning preprocessing.

Developed by Engr. Tufail Mabood for reproducible statistical analysis, data exploration,
and machine learning preprocessing in research projects.
"""

# Import all required libraries (This is developed in 3.12.3 Python)
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

# User configuration
# Ensure your cleaned dataset has no special characters in the header row
# The script automatically treats the first row as column headers
# All columns except the last non-empty column are considered predictor variables
# The last non-empty column is automatically treated as the target variable

INPUT_FILENAME = "Your Cleaned Dataset.xlsx" # In the current directory, keep your cleaned dataset and change this name
OUTPUT_DIR = Path("Univariate Multivariate Results")
IMAGE_DPI = 1200 # Very Important for Q1 Journal
FONT_NAME = "Times New Roman"
PAIRPLOT_MAX_COLS = 6   # if too many predictors, limit pairplot to top 6 corr with target
# --------------------------

def ensure_output_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def find_non_empty_columns(df: pd.DataFrame):
    non_empty = [col for col in df.columns if df[col].notna().any()]
    return non_empty

def set_font_tnr():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = FONT_NAME
    plt.rcParams['font.size'] = 10
    sns.set_style("white")

def save_hist_box(df: pd.DataFrame, col: str, outdir: Path):
    """Save histogram+kde and boxplot for one column"""
    set_font_tnr()
    series = df[col].dropna()
    if series.empty: 
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    # Histogram + KDE
    sns.histplot(series, kde=True, ax=axes[0], color="skyblue", edgecolor="black")
    axes[0].set_title(f"Histogram + KDE of {col}", fontname=FONT_NAME)
    
    # Boxplot
    sns.boxplot(y=series, ax=axes[1], color="lightgreen")
    axes[1].set_title(f"Boxplot of {col}", fontname=FONT_NAME)
    
    plt.tight_layout()
    fig.savefig(outdir / f"{col}_univariate.png", dpi=IMAGE_DPI)
    plt.close(fig)

def save_descriptive_stats(df: pd.DataFrame, outpath: Path):
    desc = df.describe(include="all").T
    desc.to_csv(outpath)
    # also save as image
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    set_font_tnr()
    fig, ax = plt.subplots(figsize=(10, max(4, 0.3*len(desc))))
    ax.axis("off")
    table = ax.table(cellText=desc.round(4).values,
                     rowLabels=desc.index,
                     colLabels=desc.columns,
                     loc="center",
                     cellLoc="center")
    for _, cell in table.get_celld().items():
        cell.set_text_props(fontname=FONT_NAME, fontsize=8)
        cell.set_edgecolor("gray")
    plt.tight_layout()
    fig.savefig(outpath.with_suffix(".png"), dpi=IMAGE_DPI, bbox_inches="tight")
    plt.close(fig)

def multivariate_regression(df: pd.DataFrame, predictors, target, outdir: Path):
    """Run OLS regression and save summary + coefficients"""
    X = df[predictors]
    y = df[target]
    X = sm.add_constant(X)
    model = sm.OLS(y, X, missing="drop").fit()
    
    # Save regression summary text
    with open(outdir / "ols_regression_summary.txt", "w") as f:
        f.write(model.summary().as_text())
    
    # Save coefficients table
    coef_df = pd.DataFrame({
        "Coefficient": model.params,
        "StdErr": model.bse,
        "t": model.tvalues,
        "pValue": model.pvalues
    })
    coef_df.to_csv(outdir / "ols_coefficients.csv")
    
    # Save coefficients as image
    set_font_tnr()
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3*len(coef_df))))
    ax.axis("off")
    tbl = ax.table(cellText=coef_df.round(4).values,
                   rowLabels=coef_df.index,
                   colLabels=coef_df.columns,
                   loc="center",
                   cellLoc="center")
    for _, cell in tbl.get_celld().items():
        cell.set_text_props(fontname=FONT_NAME, fontsize=8)
        cell.set_edgecolor("gray")
    plt.tight_layout()
    fig.savefig(outdir / "ols_coefficients.png", dpi=IMAGE_DPI, bbox_inches="tight")
    plt.close(fig)

def main():
    print("Running univariate & multivariate analysis...")
    
    INPUT = Path.cwd() / INPUT_FILENAME
    if not INPUT.exists():
        raise FileNotFoundError(f"{INPUT} not found.")

    df_raw = pd.read_excel(INPUT, header=0)
    non_empty_cols = find_non_empty_columns(df_raw)
    target_col = non_empty_cols[-1]
    predictor_cols = non_empty_cols[:-1]
    
    print("Target column:", target_col)
    print("Predictors:", predictor_cols)
    
    df = df_raw[predictor_cols + [target_col]].copy()
    ensure_output_dir(OUTPUT_DIR)
    
    # Select numeric only for analysis
    df_num = df.select_dtypes(include=[np.number]).copy()
    
    # ---------------- Univariate Analysis ----------------
    uni_dir = OUTPUT_DIR / "univariate"
    ensure_output_dir(uni_dir)
    
    print("Performing univariate analysis...")
    for col in df_num.columns:
        save_hist_box(df_num, col, uni_dir)
    
    save_descriptive_stats(df_num, uni_dir / "descriptive_statistics.csv")
    
    # ---------------- Multivariate Analysis ----------------
    multi_dir = OUTPUT_DIR / "multivariate"
    ensure_output_dir(multi_dir)
    
    print("Performing multivariate analysis...")
    
    # Pairplot (limit to top correlated predictors if too many)
    corrs = df_num.corr()[target_col].abs().sort_values(ascending=False)
    top_predictors = [c for c in corrs.index if c != target_col][:PAIRPLOT_MAX_COLS]
    if top_predictors:
        plot_cols = top_predictors + [target_col]
        set_font_tnr()
        g = sns.pairplot(df_num[plot_cols], diag_kind="kde", corner=True)
        g.fig.suptitle(f"Pairplot of top predictors vs {target_col}", fontname=FONT_NAME, y=1.02)
        g.savefig(multi_dir / "pairplot.png", dpi=IMAGE_DPI)
        plt.close(g.fig)
    
    # OLS Regression
    if predictor_cols:
        predictors_used = [c for c in predictor_cols if c in df_num.columns]
        if predictors_used and target_col in df_num.columns:
            multivariate_regression(df_num, predictors_used, target_col, multi_dir)
    
    print("All results saved in:", OUTPUT_DIR.resolve())

if __name__ == "__main__":
    main()
