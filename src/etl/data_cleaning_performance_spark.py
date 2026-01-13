from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from data_cleaning_grouping_functions import (clean_standard_numeric_column_spark, clean_binary_flag_spark, 
    clean_standard_categorical_column_spark, clean_date_dependent_ltv_ratio_spark, clean_standard_datetime_column_spark,
    clean_variable_string_categorical_column_spark, clean_financial_cost_column_spark
)

# Function for dropping unnecessary columns from Performance dataset
def drop_performance_columns_spark(df_spark: DataFrame) -> DataFrame:
    print("Dropping unnecessary columns from Performance Data...")

    cols_to_drop = [
        'MONTHS_REMAINING_TO_LEGAL_MATURITY', 'DEFECT_SETTLEMENT_DATE', 'MI_RECOVERIES', 'NET_SALE_PROCEEDS', 'NON_MI_RECOVERIES', 'TOTAL_EXPENSES', 'LEGAL_COSTS', 'MAINTENANCE_AND_PRESERVE_COSTS', 'TAXES_AND_INSURANCE', 'MISC_EXPENSES', 'ACTUAL_LOSS_CALC', 'ZERO_BALANCE_REMOVAL_UPB', 'INTEREST_BEARING_UPB'
    ]

    existing_cols_to_drop = [col_name for col_name in cols_to_drop if col_name in df_spark.columns]

    if existing_cols_to_drop:
        print(f"    Dropping columns: {existing_cols_to_drop}")
        df_spark = df_spark.drop(*existing_cols_to_drop)
    else:
        print("    No columns to drop.")

    return df_spark

# Function for validating LOAN_SEQUENCE_NUMBER
def validate_loan_sequence_number_spark(df_spark: DataFrame) -> DataFrame:
    loan_seq_col = 'LOAN_SEQUENCE_NUMBER'

    if loan_seq_col not in df_spark.columns:
        print(f"Warning: {loan_seq_col} not found in DataFrame. Skipping validation.")
        return df_spark
    
    print("Validating LOAN_SEQUENCE_NUMBER (Primary Key)...")

    # 1. Ensure LOAN_SEQUENCE_NUMBER is a string for consistent validation and joining
    df_spark = df_spark.withColumn(loan_seq_col, col(loan_seq_col).cast(StringType()))
    print(f"    Cast {loan_seq_col} to StringType.")

    # 2. Check for Nulls and drop corresponding Nulls
    initiaL_rows_count = df_spark.count()
    null_count_before_drop = df_spark.filter(col(loan_seq_col).isNull()).count()

    if null_count_before_drop > 0:
        print(f"    Found {null_count_before_drop} Nulls in {loan_seq_col}. Dropping corresponding rows.")
        df_spark = df_spark.dropna(subset=[loan_seq_col])
        print(f"    Remaining rows after dropping nulls: {df_spark.count()}")
    else:
        print(f"    No Nulls found in {loan_seq_col}. Skipping drop.")

    print(f"{loan_seq_col} validation complete.")
    print(f"Rows before validation: {initiaL_rows_count}, Remaining rows after validation: {df_spark.count()}")
    return df_spark

# --- Financial Cost Cleaning Pattern Group ---

# Function for cleaning 'CUMULATIVE_MOD_COST' column
def clean_cummulative_mod_cost_spark(df_spark: DataFrame) -> DataFrame:
    return clean_financial_cost_column_spark(df_spark, 'CUMULATIVE_MOD_COST', 'CUMULATIVE_MOD_COST_IS_MODIFIED')

# Function for cleaning 'DELINQUENT_ACCRUED_INTEREST' column
def clean_delinquent_accrued_interest_spark(df_spark: DataFrame) -> DataFrame:
    return clean_financial_cost_column_spark(df_spark, 'DELINQUENT_ACCRUED_INTEREST', 'HAS_DELINQUENT_INTEREST')

# Function for cleaning 'CURRENT_MONTH_MOD_COST' column
def clean_current_month_mod_cost_spark(df_spark: DataFrame) -> DataFrame:
    return clean_financial_cost_column_spark(df_spark, 'CURRENT_MONTH_MOD_COST', 'CURRENT_MONTH_IS_MODIFIED')