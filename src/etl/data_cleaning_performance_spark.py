from typing import final
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, substring, length, concat_ws, to_date, sum as spark_sum, count as spark_count, mode as spark_mode, lpad, lit, count, when, median,
    upper, trim, expr
)
from pyspark.sql.types import IntegerType, DateType, StringType, FloatType, ByteType, ShortType
import pandas as pd
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

# Function for cleaning MONTHLY_REPORTING_PERIOD column
def clean_monthly_reporting_period_spark(df_spark: DataFrame) -> DataFrame:

    if 'MONTHLY_REPORTING_PERIOD' not in df_spark.columns:
        print("Warning: MONTHLY_REPORTING_PERIOD not found.")
        return df_spark

    # Idempotency check
    current_type = df_spark.schema['MONTHLY_REPORTING_PERIOD'].dataType
    if isinstance(current_type, DateType):
        print("    MONTHLY_REPORTING_PERIOD is already DateType. Skipping.")
        return df_spark

    # Standardize and Convert
    df_spark = df_spark.withColumn(
        'MONTHLY_REPORTING_PERIOD',
        to_date(trim(col('MONTHLY_REPORTING_PERIOD').cast(StringType())), 'yyyyMM')
    )

    # Drop NULLs
    initial_count = df_spark.count()
    df_spark = df_spark.dropna(subset=['MONTHLY_REPORTING_PERIOD'])
    final_count = df_spark.count()

    if initial_count != final_count:
        print(f"    Dropped {initial_count - final_count} rows with invalid dates.")

    print("MONTHLY_REPORTING_PERIOD conversion complete.")
    return df_spark

# --- Standard Numeric Cleaning Pattern Group ---

# Function for cleaning 'CURRENT_ACTUAL_UPB' column
def clean_current_actual_upb_spark(df_spark: DataFrame) -> DataFrame:
    return clean_standard_numeric_column_spark(df_spark, 'CURRENT_ACTUAL_UPB', None, 0, 3000000, IntegerType())

# Function for cleaning 'LOAN_AGE' column
def clean_loan_age_spark(df_spark: DataFrame) -> DataFrame:
    return clean_standard_numeric_column_spark(df_spark, 'LOAN_AGE', None, 0, 600, ShortType())

# Function for cleaning 'CURRENT_INTEREST_RATE' column
def clean_current_interest_rate_spark(df_spark: DataFrame) -> DataFrame:
    return clean_standard_numeric_column_spark(df_spark, 'CURRENT_INTEREST_RATE', 0, 0.5, 20.0, FloatType())

# Function for cleaning 'CURRENT_NON_INTEREST_BEARING_UPB' column
def clean_non_interest_upb_spark(df_spark: DataFrame) -> DataFrame:
    return clean_standard_numeric_column_spark(df_spark, 'CURRENT_NON_INTEREST_BEARING_UPB', None, 0, 3000000, IntegerType())

# Function for cleaning 'ELTV' column
def clean_eltv_spark(df_spark:DataFrame) -> DataFrame:
    return clean_standard_numeric_column_spark(df_spark, 'ELTV', 999, 1, 200, ShortType())

# --- Binary Flag Cleaning Pattern Group ---

# Function for cleaning 'DELINQUENCY_DUE_TO_DISASTER' column (must be done after monthly reporting period cleaning)
def clean_delinquency_disaster_spark(df_spark: DataFrame) -> DataFrame:
    if 'DELINQUENCY_DUE_TO_DISASTER' not in df_spark.columns:
        print("Warning: DELINQUENCY_DUE_TO_DISASTER not found. Skipping.")
        return df_spark

    print("Cleaning DELINQUENCY_DUE_TO_DISASTER column...")

    df_spark = clean_binary_flag_spark(df_spark, 'DELINQUENCY_DUE_TO_DISASTER', {'Y': 1})

    indicator_col = 'DISASTER_FLAG_MISSING'
    cutoff_date = '2014-01-01'

    if indicator_col not in df_spark.columns:
        df_spark = df_spark.withColumn(
            indicator_col,
            when(col('MONTHLY_REPORTING_PERIOD') < lit(cutoff_date), lit(1)).otherwise(lit(0)).cast(ByteType())
        )
        print(f"    Created {indicator_col} based on {cutoff_date} cutoff.")

    return df_spark

# --- Standard Categorical Cleaning Pattern Group ---

# Function for cleaning 'MOD_FLAG' column
def clean_mod_flag_spark(df_spark: DataFrame) -> DataFrame:
    return clean_standard_categorical_column_spark(df_spark, 'MOD_FLAG', {'Y': 'CURRENT_PERIOD_MOD', 'P': 'PRIOR_PERIOD_MOD'}, 'NOT_MODIFIED')

# Function for cleaning 'ZERO_BALANCE_CODE' column
def clean_zero_balance_code_spark(df_spark: DataFrame) -> DataFrame:
    return clean_standard_categorical_column_spark(df_spark, 'ZERO_BALANCE_CODE', {'01': 'PREPAID_OR_MATURED', '02': 'THIRD_PARTY_SALE',
                                                                                   '03': 'SHORT_SALE_OR_CHARGE_OFF', '09': 'REO_DISPOSITION',
                                                                                   '15': 'WHOLE_LOAN_SALE', '16': 'REPERFORMING_LOAN_SECURITIZATION',
                                                                                   '96': 'DEFECT_PRIOR_TO_OTHER_EVENT'}, 'NO_ZERO_BALANCE_CODE')

# Function for cleaning 'STEP_MOD_FLAG' column
def clean_step_mod_flag_spark(df_spark: DataFrame) -> DataFrame:
    return clean_standard_categorical_column_spark(df_spark, 'STEP_MOD_FLAG', {'Y': 'STEP_MOD', 'N': 'NON_STEP_MOD'}, 'LOAN_NOT_MODIFIED')

# Function for cleaning 'PAYMENT_DEFERRAL_FLAG' column
def clean_payment_deferral_flag_spark(df_spark: DataFrame) -> DataFrame:
    return clean_standard_categorical_column_spark(df_spark, 'PAYMENT_DEFERRAL_FLAG', {'Y': 'CURRENT_PERIOD', 'P': 'PRIOR_PERIOD'}, 'NOT_PAYMENT_DEFERRAL')

# Function for cleaning 'BORROWER_ASSISTANCE_STATUS_CODE' column
def clean_borr_assist_code_spark(df_spark: DataFrame) -> DataFrame:
    return clean_standard_categorical_column_spark(df_spark, 'BORROWER_ASSISTANCE_STATUS_CODE', {'F': 'FORBEARANCE', 'R': 'REPAYMENT',
                                                                                                 'T': 'TRIAL_PERIOD'}, 'NO_BORR_ASSIST_CODE')

# --- Financial Cost Cleaning Pattern Group ---

# Function for cleaning 'CUMULATIVE_MOD_COST' column
def clean_cumulative_mod_cost_spark(df_spark: DataFrame) -> DataFrame:
    return clean_financial_cost_column_spark(df_spark, 'CUMULATIVE_MOD_COST', 'CUMULATIVE_MOD_COST_IS_MODIFIED')

# Function for cleaning 'DELINQUENT_ACCRUED_INTEREST' column
def clean_delinquent_accrued_interest_spark(df_spark: DataFrame) -> DataFrame:
    return clean_financial_cost_column_spark(df_spark, 'DELINQUENT_ACCRUED_INTEREST', 'HAS_DELINQUENT_INTEREST')

# Function for cleaning 'CURRENT_MONTH_MOD_COST' column
def clean_current_month_mod_cost_spark(df_spark: DataFrame) -> DataFrame:
    return clean_financial_cost_column_spark(df_spark, 'CURRENT_MONTH_MOD_COST', 'CURRENT_MONTH_IS_MODIFIED')

# --- Unique Cleaning Patterns ---

# Function for cleaning 'ZERO_BALANCE_EFFECT_DATE' column
def clean_zero_balance_effect_date_spark(df_spark: DataFrame, 
                                         column_name: str = "ZERO_BALANCE_EFFECT_DATE", 
                                         date_format_str: str = "yyyyMM") -> DataFrame:
    if column_name not in df_spark.columns:
        print(f"Warning: {column_name} not found.")
        return df_spark

    print(f"Applying specialized cleaning to {column_name}...")

    # CHECK THE TYPE: If it's already a date, don't try to parse it again!
    current_type = df_spark.schema[column_name].dataType
    if isinstance(current_type, DateType):
        print(f"    {column_name} is already DateType. Skipping conversion logic.")
        return df_spark

    # 1. Standardize Whitespace/Empty Strings to NULL
    # This ensures ' ' is treated exactly the same as a missing value
    df_spark = df_spark.withColumn(
        column_name, 
        when(trim(col(column_name).cast(StringType())) == "", lit(None))
        .otherwise(col(column_name))
    )

    # 2. Create the Indicator Column
    # 1 = Still Active (No zero balance date yet), 0 = Terminated
    is_missing_col_name = f"{column_name}_IS_MISSING"
    df_spark = df_spark.withColumn(
        is_missing_col_name, 
        when(col(column_name).isNull(), 1).otherwise(0).cast(ByteType())
    )
    print(f"  Created {is_missing_col_name} (1 = Active/Missing, 0 = Terminated).")

    # 3. Convert to DateType
    df_spark = df_spark.withColumn(
        column_name, 
        to_date(col(column_name).cast(StringType()), date_format_str)
    )

    print(f"  Final conversion of {column_name} to DateType complete.")
    return df_spark

# Function for cleaning 'CURRENT_DELINQUENCY_STATUS' column
def clean_delinquency_status_spark(df_spark: DataFrame) -> DataFrame:
    del_status_col = 'CURRENT_DELINQUENCY_STATUS'

    if del_status_col not in df_spark.columns:
        print(f"Warning: {del_status_col} not found. Skipping.")
        return df_spark 
    
    # 0. Idempotency check
    if isinstance(df_spark.schema[del_status_col].dataType, ByteType):
        print(f"{del_status_col} is already cleaned (ByteType). Skipping.")
        return df_spark
    
    print(f"Cleaning {del_status_col} column...")

    # 1. Standardize string format (strip whitespace, make uppercase)
    df_spark = df_spark.withColumn(
        del_status_col,
        upper(trim(col(del_status_col).cast(StringType())))
    )

    # 2. Create IS_RA indicator column
    df_spark = df_spark.withColumn(
        f"{del_status_col}_IS_RA",
        when(col(del_status_col) == 'RA', lit(1)).otherwise(lit(0)).cast(ByteType())
    )   

    # 3. Handle numeric conversion and initial missing flag
    df_spark = df_spark.withColumn(
        'numeric_del_temp',
        expr(f"try_cast({del_status_col} AS INT)")
    )

    # 4. Create IS_MISSING indicator column.
    # Logic: We consider it missing if the original was NULL or non-numeric. However, if it was 'RA', it was not missing
    df_spark = df_spark.withColumn(
        f"{del_status_col}_IS_MISSING",
        when(
            (col('numeric_del_temp').isNull()) & (col(f"{del_status_col}_IS_RA") == 0),
            lit(1)
        ).otherwise(lit(0)).cast(ByteType())
    )

    # 5. Impute with 0 and clip outliers
    max_del_cap = 18
    df_spark = df_spark.withColumn(
        del_status_col,
        when(col('numeric_del_temp').isNull(), lit(0)) # Impute
        .when(col('numeric_del_temp') > max_del_cap, lit(max_del_cap)) # Cap Max
        .when(col('numeric_del_temp') < 0, lit(0)) # Cap Min
        .otherwise(col('numeric_del_temp')).cast(ByteType())
    )

    # 6. Drop temp column
    df_spark = df_spark.drop('numeric_del_temp')

    print(f"Finished cleaning {del_status_col} column.")
    return df_spark

# Function for cleaning 'DDLPI' column
def clean_ddlpi_spark(df_spark: DataFrame) -> DataFrame:
    column_name = 'DDLPI'
    if column_name not in df_spark.columns:
        print(f"Warning: {column_name} not found. Skipping.")
        return df_spark
    
    # 0. Idempotency Check
    if isinstance(df_spark.schema[column_name].dataType, DateType):
        print(f"Warning: {column_name} is already a date. Skipping.")
        return df_spark
    
    print(f"Cleaning {column_name} column...")

    # 1. Standardize string format (strip whitespace, make uppercase) and handle empty strings
    df_spark = df_spark.withColumn(
        column_name,
        when(trim(upper(col(column_name).cast(StringType()))) == '', lit(None))
        .otherwise(trim(upper(col(column_name).cast(StringType()))))
    )

    # 2. Create IS_MISSING indicator column
    df_spark = df_spark.withColumn(
        f"{column_name}_IS_MISSING",
        when(col(column_name).isNull(), lit(1))
        .otherwise(lit(0)).cast(ByteType())
    )

    # 3. Convert YYYYMM to DateType
    df_spark = df_spark.withColumn(
        column_name,
        to_date(col(column_name), 'yyyyMM')
    )
    print(f"    Final conversion of {column_name} to DateType complete.")
    return df_spark

# --- Master Performance Cleaning ---
def clean_performance_spark(raw_df: DataFrame) -> DataFrame:
    # Drop columns
    df_spark = drop_performance_columns_spark(raw_df)
    # Validate LOAN_SEQUENCE_NUMBER
    df_spark = validate_loan_sequence_number_spark(df_spark)
    # Clean MONTHLY_REPORTING_PERIOD column (must be done before some other columns)
    df_spark = clean_monthly_reporting_period_spark(df_spark)
    # Clean all other columns
    print("Beginning cleaning of performance dataset")
    # Standard Numeric Cleaning Pattern Group
    df_spark = clean_current_actual_upb_spark(df_spark)
    df_spark = clean_loan_age_spark(df_spark)
    df_spark = clean_current_interest_rate_spark(df_spark)
    df_spark = clean_non_interest_upb_spark(df_spark)
    df_spark = clean_eltv_spark(df_spark)
    # Binary Flag Cleaning Pattern Group
    df_spark = clean_delinquency_disaster_spark(df_spark)
    # Standard Categorical Cleaning Pattern Group
    df_spark = clean_mod_flag_spark(df_spark)
    df_spark = clean_zero_balance_code_spark(df_spark)
    df_spark = clean_step_mod_flag_spark(df_spark)
    df_spark = clean_payment_deferral_flag_spark(df_spark)
    df_spark = clean_borr_assist_code_spark(df_spark)
    # Financial Cost Cleaning Pattern Group
    df_spark = clean_cumulative_mod_cost_spark(df_spark)
    df_spark = clean_delinquent_accrued_interest_spark(df_spark)
    df_spark = clean_current_month_mod_cost_spark(df_spark)
    # Unique Cleaning Patterns
    df_spark = clean_zero_balance_effect_date_spark(df_spark)
    df_spark = clean_delinquency_status_spark(df_spark)
    df_spark = clean_ddlpi_spark(df_spark)

    return df_spark
