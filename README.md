# 📊 Analysis Overview

This notebook performs a complete machine learning pipeline for **flood risk prediction** using geospatial raster features. The analysis is structured into **6 major parts**.

---

## Part 1 – Data Loading & Exploration

The dataset (`Training_Data_16k.csv`, ~16k samples) is loaded and explored. Features are categorized into **continuous** (Elevation, Slope, Rainfall, TWI, SPI, etc.), **categorical** (LULC, Geomorphology, Lithology, Soil), and one **circular** feature (Aspect). The binary target variable represents flood risk (**0 = no flood, 1 = flood**)

---

## Part 2 – Data Preprocessing

Invalid/special values (`-128`, `-1`, `NA`, `65535`) are replaced with NaN and rows with missing values are removed. Features are then processed based on their type:

- **Continuous** → Z-score normalization (`StandardScaler`)
- **Categorical** → One-hot encoding (with `drop_first` to avoid the dummy variable trap)
- **Circular (Aspect)** → Sin/Cos transformation to preserve angular continuity

All processed feature groups are concatenated into a final dataset and saved to CSV files, along with visualizations of distributions before and after transformation.

---

## Part 3 – Correlation Analysis

Three correlation methods are applied:

- **Pearson correlation** between continuous features (flagging pairs with |r| > 0.7)
- **Point-Biserial correlation** between all features and the binary target — top 20 most correlated features visualized
- **Chi-Square test + Cramér's V** for categorical features vs. the target, with statistical significance testing

---

## Part 4 – Collinearity & Feature Selection

Collinearity analysis identifies highly correlated feature pairs (|r| > 0.8) and computes **VIF (Variance Inflation Factor)** for all features, categorizing them as **no / moderate / high / severe** multicollinearity.

Feature selection is performed using **four independent methods**:

- **Mutual Information** — measures dependency between each feature and the target
- **Random Forest Feature Importance** — built-in importance from a trained RF model (100 trees, max depth 10)
- **Permutation Importance** — measures accuracy drop when each feature is randomly shuffled
- **ANOVA F-Statistic** — tests statistical difference in feature means between classes

All four methods are normalized and averaged into a **Combined Ranking**, producing recommended feature sets of **10, 15, and 20** top features. A **consensus analysis** identifies features that appear in the top 10 across multiple methods.

---

## Part 5 – Accuracy Assessment

**Twelve different feature sets** are benchmarked (Top 10/20 from each individual method + combined sets + all features as baseline). For each set, a **Random Forest classifier** is trained and evaluated using:

- **Accuracy, Precision, Recall, F1-Score**
- **5-fold Cross-Validation**

Results are compared to identify the best trade-off between the number of features and model performance.

---

## Part 6 – Hyperparameter Tuning

The best-performing feature set from Part 5 is used to fine-tune the Random Forest model in two stages:

1. **Randomized Search** (50 combinations, 3-fold CV) — broad exploration of the hyperparameter space
2. **Grid Search** (5-fold CV) — fine-grained search around the best parameters from step 1

The baseline, randomized search, and grid search models are compared across all metrics. A **confusion matrix** and full **classification report** are generated for the best model, and the final optimized parameters are saved.

---

## 📁 Output Files

The analysis produces a comprehensive set of saved files including processed datasets, feature ranking CSVs, accuracy assessment results, best model parameters, and **15+ visualization PNGs** covering distributions, correlations, VIF, feature importances, and model comparisons.
