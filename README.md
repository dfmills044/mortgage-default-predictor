# 🏦 Mortgage Default Risk Pipeline: PySpark ETL Migration

## 📌 Project Overview
This repository contains a high-volume **ETL and Feature Engineering pipeline** designed to process the **Freddie Mac Single Family Loan-Level Dataset**.

The goal is to prepare historical loan performance data for a machine learning model that predicts default risk. This project leverages **distributed data processing** via PySpark to handle tens of millions of records (2023–2026 window), transforming raw monthly logs into a flattened, model-ready training set.

## 🏗 Architecture & Engineering Strategy
**Current Status:** 🧪 **Feature Engineering & Model Selection**

The pipeline has successfully moved beyond basic cleaning into sophisticated longitudinal aggregation and cross-dataset merging.

### Key Accomplishments:
* ✅ **Origination Pipeline:** Full schema enforcement, validation of primary keys, and deriving chronological markers for tens of millions of loans.
* ✅ **Performance Pipeline:** Complex multi-stage cleaning and validation of monthly credit logs, including the handling of "impossible" delinquency jumps and UPB resets.
* ✅ **Censored Aggregation Logic:** Developed a "point-in-time" feature generator that masks data at the first instance of default to prevent data leakage.
* ✅ **Look-Ahead Labeling:** Created a robust `LABEL_DEFAULT_6MO` target that predicts if a loan will fail within the next two quarters, significantly increasing the model's predictive value.
* ✅ **Grand Join:** Successfully implemented the master merge logic that unifies static Origination traits with behavioral Performance features.

## 📂 Repository Structure
* `src/etl/common_utils.py`: Shared UDFs for postal code validation and categorical mapping.
* `src/etl/data_cleaning_grouping_functions.py`: Standardized cleaning patterns (Numeric, Binary, Categorical) used across all pipelines.
* `src/etl/data_cleaning_originations_spark.py`: Logic for cleaning and schema enforcement of the static Origination dataset.
* `src/etl/data_cleaning_performance_spark.py`: Monthly log cleaning, handling null-imputation for ELTV, interest rates, and financial flags.
* `src/etl/data_aggregation_performance_spark.py`: The "Engine" — uses a censored mask and temporal buffer to aggregate 50+ features per loan.
* `src/etl/data_cleaning_agg_merge_spark.py`: The Master Pipeline script that orchestrates the end-to-end flow from raw files to the final training set.

## 🚀 Key Technical Decisions
### 1. Point-in-Time Feature Masking & Temporal Buffer
To ensure the model learns to predict default rather than simply observing it, we implement a **Censored Mask**. We identify the first month of default and filter features prior to that event. Furthermore, we implement a **6-month temporal buffer** at the end of the dataset to ensure every "active" loan has a fair look-ahead period to prove its status.

### 2. Reset-Proof Chronological Age
Standard "Loan Age" resets during modifications. We implemented a `CHRONO_AGE` calculation based on the observation start date to maintain a consistent survival timeline across the 2023–2026 data window.

### 3. Modular Cleaning Patterns
By abstracting cleaning logic into reusable "Patterns" (e.g., `clean_standard_numeric_column_spark`), the pipeline ensures that missing value indicators, median imputation, and outlier clipping are applied identically across the entire feature space.

## 🔮 Roadmap
* ✅ **Phase 1:** PySpark Environment Setup & Origination Data Cleaning.
* ✅ **Phase 2:** Performance ETL & "Censored" Aggregator (50+ behavioral features).
* ✅ **Phase 3:** The Grand Join (Merging Origination + Performance + Labels).
* 🚧 **Phase 4:** Feature Engineering (Derived Ratios like Interest Rate Shock and Equity Stress).
* ⏳ **Phase 5:** Model Development (XGBoost/LightGBM) & Hyperparameter Tuning.
* ⏳ **Phase 6:** Backtesting & Out-of-Time Validation (2025/2026 samples).

## ✍️ Author
**Duncan Mills**
* *Product Analyst & Quantitative Finance Enthusiast*
* **LinkedIn:** [linkedin.com/in/duncan-f-mills](https://linkedin.com/in/duncan-f-mills)