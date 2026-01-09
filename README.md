# 🏦 Mortgage Default Risk Pipeline: PySpark ETL Migration

## 📌 Project Overview
This repository contains a high-volume **ETL and Feature Engineering pipeline** designed to process the **Freddie Mac Single Family Loan-Level Dataset**.

The goal is to prepare historical loan performance data for a machine learning model that predicts default risk. Given the dataset size (tens of millions of records across Origination and Performance logs), this project focuses on **distributed data processing** to overcome local memory constraints.

## 🏗 Architecture & The "Pivot"
**Current Status:** 🔄 *Migrating from Pandas to PySpark*

I originally prototyped this pipeline using **Pandas**. However, as the dataset grew to include multiple historical quarters (>20GB), local memory (RAM) became a bottleneck.

I am currently refactoring the ingestion layer to **PySpark (Databricks)** to enable:
* **Distributed Processing:** Handling the 31-column Origination and 19-column Performance datasets in parallel.
* **Scalability:** Allowing the model to ingest decades of historical data without performance degradation.

## 📂 Repository Structure
* `src/etl/data_cleaning_originations_spark.py`: **[Active]** PySpark logic for cleaning and schema enforcement of the Origination dataset.
* `src/etl/data_cleaning_performance_spark.py`: **[In Progress]** Skeleton logic for the monthly performance logs.
* `legacy/`: The original local processing scripts (archived).

## 🚀 Key Technical Decisions
### 1. Handling High-Cardinality Data
The Originations dataset contains mixed types (Credit Scores, DTI, Zip Codes). The PySpark pipeline enforces strict schema validation to reject malformed rows early in the ingestion process.

### 2. Null Strategy
Financial data is notoriously messy. This pipeline implements specific imputation strategies:
* **Credit Scores:** Filtered for valid ranges (300-850).
* **First-Time Homebuyer:** Imputed 'Unknown' values based on loan purpose logic.

## 🔮 Roadmap
* ✅ **Phase 1:** PySpark Environment Setup & Origination Data Cleaning.
* 🚧 **Phase 2:** Performance Data ETL & Key Generation (merging heavily on `LOAN_SEQUENCE_NUMBER`).
* ⏳ **Phase 3:** Feature Engineering (Creating `delinquency_status` targets).
* ⏳ **Phase 4:** ML Model Training (Logistic Regression / Gradient Boosting).

## ✍️ Author
**Duncan Mills**
* *Product Analyst & Quantitative Finance Enthusiast*
* **LinkedIn:** linkedin.com/in/duncan-f-mills