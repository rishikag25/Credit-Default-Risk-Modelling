# Credit Default Risk Modeling

An advanced quantitative framework designed to estimate transaction-level credit default probabilities and evaluate portfolio Expected Loss (EL) metrics. The system implements robust missing data imputation, corrects extreme class imbalances via synthetic sampling, and isolates predictive risk signatures using optimized gradient-boosted ensembles.

---

##  Pipeline Architecture

* **Data Synthesis:** Aggregation of credit profiles, missing asset records, and institutional transaction data.
* **Hypothesis Testing:** Verification of cash-transaction risk dynamics mapped against structural credit distributions.
* **Class Imbalance Rectification:** Target class balancing using Synthetic Minority Over-sampling Technique (SMOTE).
* **Feature Engineering:** Normalization and isolation of input metrics within cross-validation folds to prevent forward-looking data leakage.
* **Ensemble Optimization:** Hyperparameter tuning of an optimal XGBoost architecture to calculate precise default probabilities.
* **Loss Quantification:** Conversion of validation outputs into portfolio Expected Loss metrics.

---

##  Project Overview

In institutional credit lending, default events represent a significant driver of unexpected financial losses. Because default instances are statistically rare compared to healthy accounts, building predictive models requires handling severe class imbalances and incomplete profile data. This project delivers an end-to-end evaluation engine that handles messy underwriting sheets and outputs precise probability metrics to quantify structural risk exposure.

---

##  Data & Feature Engineering

### 1. Dataset Description
* **Data Scope:** Synthesized complex credit applicant profiles containing missing asset data alongside granular transaction histories.
* **Domain Elements:** Tracks asset valuations, transaction frequency, repayment logs, and historical credit usage markers.

### 2. Preprocessing & Leakage Controls
* **Hypothesis Imputation:** Resolved missing asset profiles by verifying an explicit cash-transaction risk hypothesis against baseline credit distributions via data analysis.
* **Imbalance Resolution:** Deployed **SMOTE (Synthetic Minority Over-sampling Technique)** to mathematically generate synthetic minority class examples, preventing the classifier from developing a majority-class accuracy bias.
* **Leakage Containment:** Enforced strict variable scaling and transformation protocols inside isolated cross-validation folds, completely eliminating forward-looking data leakage.

---

##  Methodology & Modeling Approach

### 1. High-Performance Ensemble Modeling
* **Core Algorithm:** Tuned and deployed an optimal **XGBoost (Extreme Gradient Boosting)** classifier to map complex, non-linear interactions between transactional liquidity and applicant default risk.
* **Hyperparameter Control:** Regulated learning rates, maximum tree depth, and subsample ratios to prevent overfitting and maximize model generalization.

### 2. Credit Risk Analytics & Validation
* **Model Validation:** Secured an **ROC-AUC of 0.670**, striking a mathematically viable trade-off between default detection and false alarm rates on highly skewed credit distributions.
* **Downstream Valuation:** Utilized the resulting classification probabilities to compute highly precise portfolio **Expected Loss** metrics.

---

##  Tech Stack & Tooling

* **Core Programming Language:** Python
* **Data Wrangling & Rebalancing:** NumPy, Pandas, Imbalanced-learn (SMOTE), Scikit-learn
* **Gradient Boosting Core:** XGBoost Library
* **Visualization Matrix:** Matplotlib, Seaborn

---

##  Data Source & Risk Parameters

The framework evaluates fundamental risk parameters to isolate default probability:
* **Target Metric:** Probability of Default (PD) at the individual applicant level.
* **Risk Predictors:** Cash transaction liquidity ratios, asset-to-liability indicators, historical credit utilization markers, and repayment velocity scores.

---

##  Key Results & Deliverables

* **Imbalance Correction:** Shifted raw, heavily skewed credit matrices into mathematically balanced training sets without distorting real-world distribution signatures.
* **Risk Quantification:** Provided an operational pipeline that outputs true Expected Loss metrics, directly serving risk-mitigation decision-making.
