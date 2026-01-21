# 🏦 Mortgage Default Risk Pipeline: PySpark ETL Migration

## 📌 Project Overview
This repository contains a high-volume **ETL and Feature Engineering pipeline** designed to process the **Freddie Mac Single Family Loan-Level Dataset**.

The goal is to prepare historical loan performance data for a machine learning model that predicts default risk. This project leverages **distributed data processing** via PySpark to handle tens of millions of records (2023–2026 window), transforming raw monthly logs into a flattened, model-ready training set.

## 🏗 Architecture & Engineering Strategy
**Current Status:** 🧪 **Feature Engineering & Model Selection**

The pipeline has successfully moved beyond basic cleaning into sophisticated longitudinal aggregation. 

### Key Accomplishments:
* ✅ **Origination Pipeline:** Full schema enforcement and cleaning for static loan traits.
* ✅ **Performance Pipeline:** Complex multi-stage cleaning and validation of monthly credit logs.
* ✅ **Censored Aggregation Logic:** Developed a "point-in-time" feature generator that masks data at the first instance of default to prevent data leakage.
* ✅ **Label Generation:** Created a robust `labels_df` capturing Ground Truth (Default vs. Prepaid vs. Active) and severity metrics (Loss Amount, Chrono-Age at Default).

## 📂 Repository Structure
* `src/etl/originations_cleaning.py`: PySpark logic for cleaning and schema enforcement of the static Origination dataset.
* `src/etl/performance_cleaning.py`: Monthly log cleaning, handling null-imputation for ELTV, interest rates, and financial flags.
* `src/etl/performance_aggregation.py`: The "Engine" — uses a censored mask to aggregate 50+ features per loan (e.g., payment gaps, mod costs, equity trends).
* `legacy/`: Prototyped local Pandas scripts (archived due to memory constraints).

## 🚀 Key Technical Decisions
### 1. Point-in-Time Feature Masking
To ensure the model learns to predict default rather than simply observing it, we implement a **Censored Mask**. We identify the first month of default and filter the training features to only include data *prior* to that event, while preserving the full history for the Target Label.

### 2. Reset-Proof Chronological Age
Standard "Loan Age" resets during modifications. We implemented a `CHRONO_AGE` calculation based on the observation start date to maintain a consistent survival timeline across the 2023–2026 data window.

### 3. Missingness as a Signal
For critical features like `ELTV` (Estimated Loan-to-Value), we don't just impute; we track missingness via binary flags (`ELTV_IS_MISSING`). This allows the model to distinguish between verified equity positions and imputed estimates.

## 🔮 Roadmap
* ✅ **Phase 1:** PySpark Environment Setup & Origination Data Cleaning.
* ✅ **Phase 2:** Performance ETL & "Censored" Aggregator (50+ behavioral features).
* 🚧 **Phase 3:** The Grand Join & Feature Engineering (Merging Origination + Performance + Labels).
* ⏳ **Phase 4:** Model Development (XGBoost/LightGBM) & Hyperparameter Tuning.
* ⏳ **Phase 5:** Backtesting & Temporal Validation (Testing on 2025/2026 out-of-time samples).

## ✍️ Author
**Duncan Mills**
* *Product Analyst & Quantitative Finance Enthusiast*
* **LinkedIn:** [linkedin.com/in/duncan-f-mills](https://linkedin.com/in/duncan-f-mills)