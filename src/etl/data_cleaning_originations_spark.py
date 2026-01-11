from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, substring, length, concat_ws, to_date, sum as spark_sum, count as spark_count, mode as spark_mode, lpad, lit, count, when, udf,
    upper, trim, median
)
from pyspark.sql.types import IntegerType, DateType, StringType, ShortType, ByteType, FloatType
import pandas as pd

from data_cleaning_grouping_functions import (clean_standard_numeric_column_spark, clean_binary_flag_spark, 
    clean_standard_categorical_column_spark, clean_date_dependent_ltv_ratio_spark, clean_standard_datetime_column_spark,
    clean_variable_string_categorical_column_spark
)
from common_utils import (
        _parse_msa_code_udf, _postal_code_validator_udf, _clean_numeric_string_for_map_udf, build_categorical_map_exp
    )
import datetime

# Function for dopping unnecessary columns in origination dataset
def drop_origination_columns_spark(df_spark: DataFrame) -> DataFrame:
    print("Dropping unnecessary columns from Originations Data...")

    cols_to_drop = ['PRE_RELIEF_LOAN_NUMBER']

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
    
    # 3. Check for duplicates and drop duplicate rows
    duplicate_rows_df = df_spark.groupBy(loan_seq_col).agg(spark_count(loan_seq_col).alias("count")).filter(col("count") > 1)
    duplicate_count_before_drop = duplicate_rows_df.count()

    if duplicate_count_before_drop > 0:
        print(f"    Found {duplicate_count_before_drop} duplicate rows in {loan_seq_col}. Dropping duplicate rows.")
        df_spark = df_spark.dropDuplicates(subset=[loan_seq_col])
        print(f"    Remaining rows after dropping duplicates: {df_spark.count()}")
    else:
        print(f"    No duplicates found in {loan_seq_col}. Skipping drop.")
    
    print(f"{loan_seq_col} validation complete.")
    print(f"Rows before validation: {initiaL_rows_count}, Remaining rows after validation: {df_spark.count()}")
    return df_spark

# Function for deriving the origination date from LOAN_SEQUENCE_NUMBER, creating the 'ORIGINATION_DATE' column.
def derive_loan_origination_date(df_spark: DataFrame) -> DataFrame:
    loan_seq_col = 'LOAN_SEQUENCE_NUMBER'
    orig_date_col = 'ORIGINATION_DATE'

    if loan_seq_col not in df_spark.columns:
        print(f"Column '{loan_seq_col}' not found in DataFrame.")
        return df_spark

    print(f"Deriving {orig_date_col} from {loan_seq_col}...")

    # Ensure LOAN_SEQUENCE_NUMBER is string type for string operations
    df_spark = df_spark.withColumn(loan_seq_col, col(loan_seq_col).cast(StringType()))

    # Extract year_suffix and quarter_num, validate basic format, and handle '99' year
    # P is at index 0, YY is index 1-2, Q is index 3, and n is index 4
    df_spark = df_spark.withColumn("year_suffix_str", substring(col(loan_seq_col), 2, 2)) \
        .withColumn("quarter_num_str", substring(col(loan_seq_col), 5, 1)) \
        .withColumn("is_valid_seq_format", \
            (length(col(loan_seq_col)) >= 5) & \
            (col("year_suffix_str").rlike("^[0-9]{2}$")) & \
            (col("quarter_num_str").rlike("^[1-4]$")))
        
    # Calculate full year
    df_spark = df_spark.withColumn("full_year_num", \
        when(col("is_valid_seq_format"), \
            when(col("year_suffix_str") == lit('99'), lit(1999)) \
            .otherwise(lit(2000) + col("year_suffix_str").cast(IntegerType()))) \
        .otherwise(lit(None).cast(IntegerType())))
    
    # Calculate month from quarter_num
    df_spark = df_spark.withColumn("month_num", \
        when(col("is_valid_seq_format"), \
            when(col("quarter_num_str") == lit('1'), lit(1)) \
            .when(col("quarter_num_str") == lit('2'), lit(4)) \
            .when(col("quarter_num_str") == lit('3'), lit(7)) \
            .when(col("quarter_num_str") == lit('4'), lit(10))) \
        .otherwise(lit(None).cast(IntegerType())))
    
    # Construct the date string (yyyy-MM-dd format) and convert to DateType
    df_spark = df_spark.withColumn(orig_date_col, to_date(concat_ws("-", col("full_year_num"), lpad(col("month_num"), 2, '0'), lit("01")), "yyyy-MM-dd"))

    print(f"    Created '{orig_date_col}' column.")

    # Clean up intermediate columns
    df_spark = df_spark.drop("year_suffix_str", "quarter_num_str", "is_valid_seq_format", "full_year_num", "month_num")

    # Handle null dates
    initial_null_date_count = df_spark.where(col(orig_date_col).isNull()).count()
    if initial_null_date_count > 0:
        print(f"    Found {initial_null_date_count} null dates. Attempting to impute...")
        # Calculate mode for impution
        mode_date_row = df_spark.agg(spark_mode(col(orig_date_col))).first()
        mode_date = mode_date_row[0] if mode_date_row is not None else None

        if mode_date is None:
            mode_date = pd.to_datetime('2000-01-01').date()
            print(f"    Warning: Mode date was Null. Using fallback date for imputation: {mode_date}")
        else:
            if isinstance(mode_date, pd.Timestamp):
                mode_date = mode_date.date()
            print(f"    Imputing nulls in {orig_date_col} with mode date: {mode_date}")
            df_spark = df_spark.fillna(mode_date, subset=[orig_date_col])
    else:
        print(f"    No null dates found in {orig_date_col}.")
    
    print(f"{orig_date_col} derivation complete. Final datatype: {df_spark.schema[orig_date_col].dataType}")
    return df_spark

# --- Standard Numeric Cleaning Pattern Group ---

# Function for cleaning 'CREDIT_SCORE' column
def clean_credit_score_spark(df_spark: DataFrame) -> DataFrame:
    df_spark = clean_standard_numeric_column_spark(df_spark, 'CREDIT_SCORE', 9999, 300, 850, ShortType())
    return df_spark

# Function for cleaning 'MI_PERCENT' column
def clean_mi_percent_spark(df_spark: DataFrame) -> DataFrame:
    df_spark = clean_standard_numeric_column_spark(df_spark, 'MI_PERCENT', 999, 0, 55, ByteType())
    return df_spark

# Function for cleaning 'ORIGINAL_DTI_RATIO' column
def clean_original_dti_spark(df_spark: DataFrame) -> DataFrame:
    df_spark = clean_standard_numeric_column_spark(df_spark, 'ORIGINAL_DTI_RATIO', 999, 0, 65, ByteType())
    return df_spark

# Function for cleaning 'ORIGINAL_UPB' column
def clean_original_upb_spark(df_spark: DataFrame) -> DataFrame:
    df_spark = clean_standard_numeric_column_spark(df_spark, 'ORIGINAL_UPB', 0, 0, 3_000_000, IntegerType())
    return df_spark

# Function for cleaning 'ORIGINAL_INTEREST_RATE' column
def clean_original_interest_rate_spark(df_spark: DataFrame) -> DataFrame:
    df_spark = clean_standard_numeric_column_spark(df_spark, 'ORIGINAL_INTEREST_RATE', 0, 0.5, 20.0, FloatType())
    return df_spark

# --- Binary Flag Cleaning Pattern Group ---

# Function for cleaning 'FIRST_TIME_HOMEBUYER_FLAG' column
def clean_first_homebuyer_flag_spark(df_spark: DataFrame) -> DataFrame:
    df_spark = clean_binary_flag_spark(df_spark, 'FIRST_TIME_HOMEBUYER_FLAG', {'Y': 1, 'N': 0})
    return df_spark

# Function for cleaning 'PPM_FLAG' column
def clean_ppm_flag_spark(df_spark: DataFrame) -> DataFrame:
    df_spark = clean_binary_flag_spark(df_spark, 'PPM_FLAG', {'Y': 1, 'N': 0})
    return df_spark

# Function for cleaning 'SUPER_CONFORMING_FLAG' column
def clean_super_conforming_flag_spark(df_spark: DataFrame) -> DataFrame:
    df_spark = clean_binary_flag_spark(df_spark, 'SUPER_CONFORMING_FLAG', {'Y': 1})
    return df_spark

# Function for cleaning 'IO_INDICATOR' column
def clean_io_indicator_spark(df_spark: DataFrame) -> DataFrame:
    df_spark = clean_binary_flag_spark(df_spark, 'IO_INDICATOR', {'Y': 1, 'N': 0})
    return df_spark

# Function for cleaning 'RELIEF_REFI_INDICATOR' column
def clean_relief_refi_indicator_spark(df_spark: DataFrame) -> DataFrame:
    df_spark = clean_binary_flag_spark(df_spark, 'RELIEF_REFI_INDICATOR', {'Y': 1})
    return df_spark

# --- Standard Categorical Cleaning Pattern Group ---

# Function for cleaning 'NUMBER_OF_UNITS' column
def clean_num_units_spark(df_spark: DataFrame) -> DataFrame:
    df_spark = clean_standard_categorical_column_spark(df_spark, 'NUMBER_OF_UNITS', {'1': 1, '2': 2, '3': 3, '4': 4}, 'NUM_UNITS_UNKNOWN')
    return df_spark

# Function for cleaning 'OCCUPANCY_STATUS' column
def clean_occ_status_spark(df_spark: DataFrame) -> DataFrame:
    df_spark = clean_standard_categorical_column_spark(df_spark, 'OCCUPANCY_STATUS', {'P': 'PRIMARY_RESIDENCE', 'I': 'INVESTMENT_PROPERTY', 
                                                                                      'S': 'SECONDARY_HOME'}, 'OCCUPANCY_STATUS_UNKNOWN')
    return df_spark

# Function for cleaning 'CHANNEL' column
def clean_channel_spark(df_spark: DataFrame) -> DataFrame:
    df_spark = clean_standard_categorical_column_spark(df_spark, 'CHANNEL', {'R': 'RETAIL', 'B': 'BROKER', 'C': 'CORRESPONDENT', 
                                                                             'T': 'TPO_NOT_SPECIFIED'}, 'CHANNEL_UNKNOWN')
    return df_spark

# Function for cleaning 'AMORTIZATION_TYPE' column
def clean_amortization_type_spark(df_spark: DataFrame) -> DataFrame:
    df_spark = clean_standard_categorical_column_spark(df_spark, 'AMORTIZATION_TYPE', {'FRM': 'FIXED_RATE', 'ARM': 'ADJUSTABLE_RATE'}, 
                                                       'AMORTIZATION_TYPE_UNKNOWN')
    return df_spark

# Function for cleaning 'PROPERTY_STATE' column
def clean_property_state_spark(df_spark: DataFrame) -> DataFrame:
    US_STATES_AND_TERRITORIES_CANONICAL = {
    'AL': 'AL', 'AK': 'AK', 'AZ': 'AZ', 'AR': 'AR', 'CA': 'CA', 'CO': 'CO', 'CT': 'CT', 'DE': 'DE', 'DC': 'DC', 'FL': 'FL', 'GA': 'GA', 'HI': 'HI', 'ID': 'ID', 'IL': 'IL', 'IN': 'IN', 'IA': 'IA',
    'KS': 'KS', 'KY': 'KY', 'LA': 'LA', 'ME': 'ME', 'MD': 'MD', 'MA': 'MA', 'MI': 'MI', 'MN': 'MN', 'MS': 'MS', 'MO': 'MO', 'MT': 'MT', 'NE': 'NE', 'NV': 'NV', 'NH': 'NH', 'NJ': 'NJ', 'NM': 'NM',
    'NY': 'NY', 'NC': 'NC', 'ND': 'ND', 'OH': 'OH', 'OK': 'OK', 'OR': 'OR', 'PA': 'PA', 'PR': 'PR', 'RI': 'RI', 'SC': 'SC', 'SD': 'SD', 'TN': 'TN', 'TX': 'TX', 'UT': 'UT', 'VT': 'VT', 'VA': 'VA',
    'VI': 'VI', 'WA': 'WA', 'WV': 'WV', 'WI': 'WI', 'WY': 'WY', 'AS': 'AS', 'FM': 'FM', 'GU': 'GU', 'MH': 'MH', 'MP': 'MP', 'PW': 'PW'
    }

    df_spark = clean_standard_categorical_column_spark(df_spark, 'PROPERTY_STATE', US_STATES_AND_TERRITORIES_CANONICAL, 'PROPERTY_STATE_UNKNOWN')
    return df_spark

# Function for cleaning 'PROPERTY_TYPE' column
def clean_property_type_spark(df_spark: DataFrame) -> DataFrame:
    df_spark = clean_standard_categorical_column_spark(df_spark, 'PROPERTY_TYPE', {'CO': 'CONDO', 'PU': 'PUD', 'MH': 'MANUFACTURED_HOUSING', 
                                                                                 'SF': 'SINGLE_FAMILY', 'CP': 'CO-OP'}, 'PROPERTY_TYPE_UNKNOWN')
    return df_spark

# Function for cleaning 'LOAN_PURPOSE' column
def clean_loan_purpose_spark(df_spark: DataFrame) -> DataFrame:
    df_spark = clean_standard_categorical_column_spark(df_spark, 'LOAN_PURPOSE', {'P': 'PURCHASE', 'C': 'REFINANCE_CASH_OUT', 
                                                                                  'N': 'REFINANCE_NO_CASH_OUT', 'R': 'REFINANCE_NOT_SPECIFIED'}, 
                                                                                  'LOAN_PURPOSE_UNKNOWN')
    return df_spark

# Function for cleaning 'PROGRAM_INDICATOR' column
def clean_program_indicator_spark(df_spark: DataFrame) -> DataFrame:
    df_spark = clean_standard_categorical_column_spark(df_spark, 'PROGRAM_INDICATOR', {'H': 'HOME_POSSIBLE', 'F': 'HFA_ADVANTAGE', 
                                                                                       'R': 'REFI_POSSIBLE'}, 'PROGRAM_INDICATOR_UNKNOWN')
    return df_spark

# Function for cleaning 'PROPERTY_VALUATION_METHOD' column
def clean_prop_val_method_spark(df_spark: DataFrame) -> DataFrame:
    df_spark = clean_standard_categorical_column_spark(df_spark, 'PROP_VALUATION_METHOD', {'1': 'ACE_LOANS', '2': 'FULL_APPRAISAL', 
                                                                                           '3': 'OTHER_APPRAISALS', '4': 'ACE_PDR'}, 
                                                                                           'PROP_VALUATION_METHOD_UNKNOWN')
    return df_spark

# Function for cleaning 'MI_CANCEL_INDICATOR' column
def clean_mi_cancel_indicator_spark(df_spark: DataFrame) -> DataFrame:
    df_spark = clean_standard_categorical_column_spark(df_spark, 'MI_CANCEL_INDICATOR', {'Y': 'CANCELLED', 'N': 'NOT_CANCELLED', 
                                                                                         '7': 'NOT_APPLICABLE', '9': 'NOT_DISCLOSED'}, 
                                                                                         'MI_CANCEL_INDICATOR_UNKNOWN', {'NOT_APPLICABLE': '7', 
                                                                                         'NOT_DISCLOSED': '9'})
    return df_spark

# --- Date-Dependent LTV Ratio Cleaning Pattern Group ---

# Function for cleaning 'ORIGINAL_LTV' column (MUST BE DONE AFTER CREATING ORIGINATION_DATE COLUMN)
def clean_original_ltv_spark(df_spark: DataFrame) -> DataFrame:
    df_spark = clean_date_dependent_ltv_ratio_spark(df_spark, 'ORIGINAL_LTV', 'ORIGINATION_DATE', 999, (6, 105), (1, 998), ShortType())
    return df_spark

# Function for cleaning 'ORIGINAL_CLTV' column (MUST BE DONE AFTER CREATING ORIGINATION_DATE COLUMN)
def clean_original_cltv_spark(df_spark: DataFrame) -> DataFrame:
    df_spark = clean_date_dependent_ltv_ratio_spark(df_spark, 'ORIGINAL_CLTV', 'ORIGINATION_DATE', 999, (6, 200), (1, 998), ShortType())
    return df_spark

# --- Standard Datetime Cleaning Pattern Group ---

# Function for cleaning 'FIRST_PAYMENT_DATE' column
def clean_first_payment_date_spark(df_spark: DataFrame) -> DataFrame:
    df_spark = clean_standard_datetime_column_spark(df_spark, 'FIRST_PAYMENT_DATE', 'yyyyMM')
    return df_spark

# Function for cleaning 'MATURITY_DATE' column
def clean_maturity_date_spark(df_spark: DataFrame) -> DataFrame:
    df_spark = clean_standard_datetime_column_spark(df_spark, 'MATURITY_DATE', 'yyyyMM')
    return df_spark

# --- Variable String Categorical Cleaning Pattern ---

# Function for cleaning 'SELLER_NAME' column
def clean_seller_name_spark(df_spark: DataFrame) -> DataFrame:
    df_spark = clean_variable_string_categorical_column_spark(df_spark, 'SELLER_NAME', 'SELLER_NAME_UNKNOWN')
    return df_spark

# Function for cleaning 'SERVICER_NAME' column
def clean_servicer_name_spark(df_spark: DataFrame) -> DataFrame:
    df_spark = clean_variable_string_categorical_column_spark(df_spark, 'SERVICER_NAME', 'SERVICER_NAME_UNKNOWN')
    return df_spark

# --- Unique Cleaning Functions ---

# Function for cleaning 'MSA_OR_MET_DIV' column
def clean_msa_met_div_spark(df_spark: DataFrame) -> DataFrame:
    msa_col = 'MSA_OR_MET_DIV'

    if msa_col not in df_spark.columns:
        print(f"Warning: {msa_col} not found in DataFrame. Skipping cleaning.")
        return df_spark
    
    print(f"Cleaning {msa_col}...")

    # 1. Convert to string, standardize string format
    df_spark = df_spark.withColumn(msa_col, upper(trim(col(msa_col).cast(StringType()))))
    print(f"    Converted {msa_col} to StringType, trimmed, and uppercased.")

    # 2. Apply validation and formatting helper
    print(f"    Applying UDF to parse MSA codes...")
    nans_before_val_apply = df_spark.filter(col(msa_col).isNull()).count()
    df_spark = df_spark.withColumn(msa_col, _parse_msa_code_udf(col(msa_col)))
    nans_after_val_appy = df_spark.filter(col(msa_col).isNull()).count()
    invalid_format_nans = nans_after_val_appy - nans_before_val_apply
    if invalid_format_nans > 0:
        print(f"    Found {invalid_format_nans} NaNs after applying UDF.")
    else:
        print(f"    No NaNs found after applying UDF.")

    # 3. Create '_IS_MISSING' indicator column
    msa_col_is_missing = f"{msa_col}_IS_MISSING"
    if msa_col_is_missing not in df_spark.columns:
        df_spark = df_spark.withColumn(msa_col_is_missing, when(col(msa_col).isNull(), 1).otherwise(0).cast(ByteType()))
        print(f"    Created {msa_col_is_missing} column.")
        df_spark.select(msa_col_is_missing).groupBy(msa_col_is_missing).count().show(truncate=False)
    else:
        print(f"    {msa_col_is_missing} already exists in DataFrame. Skipping creation.")
    
    # 4. Fill remaining NaNs with 'NON_MSA_OR_UNKNOWN'
    final_nan_count = df_spark.filter(col(msa_col).isNull()).count()
    if final_nan_count > 0:
        print(f"    Found {final_nan_count} NaNs in {msa_col}. Filling NaNs with 'NON_MSA_OR_UNKNOWN'.")
        df_spark = df_spark.fillna('NON_MSA_OR_UNKNOWN', subset=[msa_col])
    else:
        print(f"    No NaNs found in {msa_col}. Skipping filling NaNs.")
    
    # 5. Ensure that the column is still StringType
    df_spark = df_spark.withColumn(msa_col, col(msa_col).cast(StringType()))
    print(f"    Final datatype of {msa_col}: {df_spark.schema[msa_col].dataType}")
    return df_spark

# Function for cleaning 'POSTAL_CODE' column
def clean_postal_code_spark(df_spark: DataFrame) -> DataFrame:
    postal_code_col = 'POSTAL_CODE'

    if postal_code_col not in df_spark.columns:
        print(f"{postal_code_col} not found in DataFrame. Skipping cleaning.")
        return df_spark
    
    print(f"Cleaning {postal_code_col}...")

    # 1. Ensure StringType for UDF compatibility and standardize string format
    df_spark = df_spark.withColumn(postal_code_col, trim(upper(col(postal_code_col).cast(StringType()))))
    print(f"    Converted {postal_code_col} to StringType and standardize.")

    # 2. Apply UDF
    print(f"    Applying format validation and zero-padding for {postal_code_col}...")
    nans_before_val_apply = df_spark.filter(col(postal_code_col).isNull()).count()
    df_spark = df_spark.withColumn(postal_code_col, _postal_code_validator_udf(col(postal_code_col)))
    nans_after_val_apply = df_spark.filter(col(postal_code_col).isNull()).count()
    invalid_format_count = nans_after_val_apply - nans_before_val_apply
    if invalid_format_count > 0:
        print(f"    Found {invalid_format_count} invalid format NaNs after applying UDF.")
    else:
        print(f"    No NaNs found after applying UDF. Skipping filling NaNs.")
    
    # 3. Create '_IS_MISSING' indicator column
    is_missing_col_name = f"{postal_code_col}_IS_MISSING"
    if is_missing_col_name not in df_spark.columns:
        df_spark = df_spark.withColumn(is_missing_col_name, when(col(postal_code_col).isNull(), 1).otherwise(0).cast(ByteType()))
        print(f"    Created {is_missing_col_name} column.")
        df_spark.select(is_missing_col_name).groupBy(is_missing_col_name).count().show(truncate=False)
    else:
        print(f"    {is_missing_col_name} column already exists. Skipping creating column.")

    # 4. Fill remaining NaNs with 'UNKNOWN_POSTAL_CODE'
    final_nan_count = df_spark.filter(col(postal_code_col).isNull()).count()
    if final_nan_count > 0:
        print(f"    Found {final_nan_count} NaNs after applying UDF. Filling NaNs with 'UNKNOWN_POSTAL_CODE'.")
        df_spark = df_spark.fillna('UNKNOWN_POSTAL_CODE', subset=[postal_code_col])
    else:
        print(f"    No NaNs found after applying UDF. Skipping filling NaNs.")

    # 5. Cast to StringType
    df_spark = df_spark.withColumn(postal_code_col, col(postal_code_col).cast(StringType()))
    print(f"    Final datatype of {postal_code_col}: {df_spark.schema[postal_code_col].dataType}")
    print(f"{postal_code_col} cleaning completed.")
    return df_spark

# Function for cleaning ORIGINAL_LOAN_TERM column
def clean_original_loan_term_spark(df_spark: DataFrame) -> DataFrame:
    original_loan_term_col = 'ORIGINAL_LOAN_TERM'

    if original_loan_term_col not in df_spark.columns:
        print(f"{original_loan_term_col} not found in DataFrame. Skipping cleaning.")
        return df_spark
    
    print(f"Cleaning {original_loan_term_col}...")

    # 1. Convert to numeric, coercing errors to NaN
    df_spark = df_spark.withColumn(original_loan_term_col, col(original_loan_term_col).cast(FloatType()))
    print(f"    Converted column '{original_loan_term_col}' to FloatType.")

    # 2. Convert 0 and negative values to NaN
    initial_zero_and_less_vals = df_spark.filter(col(original_loan_term_col) <= 0).count()
    if initial_zero_and_less_vals > 0:
        print(f"    Found {initial_zero_and_less_vals} zero or negative values. Converting to NaN.")
        df_spark = df_spark.withColumn(original_loan_term_col, when(col(original_loan_term_col) <= 0, None).otherwise(col(original_loan_term_col)))
    else:
        print(f"    No zero or negative values found. Skipping converting to NaN.")

    # 3. Create '_IS_MISSING' indicator column
    is_missing_col_name = f"{original_loan_term_col}_IS_MISSING"
    if is_missing_col_name not in df_spark.columns:
        df_spark = df_spark.withColumn(is_missing_col_name, when(col(original_loan_term_col).isNull(), 1).otherwise(0).cast(ByteType()))
        print(f"    Created {is_missing_col_name} column.")
        df_spark.select(is_missing_col_name).groupBy(is_missing_col_name).count().show(truncate=False)
    else:
        print(f"    {is_missing_col_name} column already exists. Skipping creating column.")

    # 4. Impute remaining NaNs with the median
    median_val_row = df_spark.agg(median(col(original_loan_term_col))).first()
    median_val = median_val_row[0] if median_val_row is not None else None

    if median_val is None:
            print(f"    Warning: Median value was Null (all values might be Null). Cannot impute. Leaving NaNs.")
    else:
        initial_nan_count = df_spark.filter(col(original_loan_term_col).isNull()).count()
        if initial_nan_count > 0:
            print(f"    Found {initial_nan_count} NaNs. Imputing with median: {median_val}")
            df_spark = df_spark.fillna(median_val, subset=[original_loan_term_col])
        else:
            print(f"    No NaNs found. Skipping imputation.")

    # 5. Clip outliers [60, 480]
    min_clip_value = 60
    max_clip_value = 480
    print(f"    Clipping {original_loan_term_col} values to range [{min_clip_value}, {max_clip_value}]")
    df_spark = df_spark.withColumn(original_loan_term_col, when(col(original_loan_term_col) < lit(min_clip_value), lit(min_clip_value)) \
                                .when(col(original_loan_term_col) > lit(max_clip_value), lit(max_clip_value)) \
                                .otherwise(col(original_loan_term_col)))
    current_min_value = df_spark.agg({original_loan_term_col: "min"}).collect()[0][0] if df_spark.count() > 0 else None
    current_max_value = df_spark.agg({original_loan_term_col: "max"}).collect()[0][0] if df_spark.count() > 0 else None
    print(f"    Clipped {original_loan_term_col} values to range [{current_min_value}, {current_max_value}]")

    # 6. Cast to IntegerType
    df_spark = df_spark.withColumn(original_loan_term_col, col(original_loan_term_col).cast(IntegerType()))
    print(f"    Final datatype of {original_loan_term_col}: {df_spark.schema[original_loan_term_col].dataType}")
    print(f"{original_loan_term_col} cleaning completed. Remaining NaNs: {df_spark.filter(col(original_loan_term_col).isNull()).count()}")
    return df_spark

# Function for cleaning 'NUMBER_OF_BORROWERS' column (MUST BE DONE AFTER CREATING ORIGINATION_DATE COLUMN)
def clean_num_borrowers_spark(df_spark: DataFrame) -> DataFrame:
    num_borr_col = 'NUMBER_OF_BORROWERS'
    orig_date_col = 'ORIGINATION_DATE'

    if num_borr_col not in df_spark.columns:
        print(f"Column {num_borr_col} not found in DataFrame. Skipping cleaning.")
        return df_spark
    
    if orig_date_col not in df_spark.columns or not isinstance(df_spark.schema[orig_date_col].dataType, DateType):
        print(f"Column {orig_date_col} not found in DataFrame or is not DateType. Skipping cleaning.")
        return df_spark
    
    print(f"Cleaning {num_borr_col}...")

    # 1. Robustly convert to numeric strings
    df_spark = df_spark.withColumn(num_borr_col, _clean_numeric_string_for_map_udf(col(num_borr_col).cast(StringType())))
    print(f"    Converted {num_borr_col} to numeric strings. Remaining NaNs: {df_spark.filter(col(num_borr_col).isNull()).count()}")

    # 2. Standardize string
    df_spark = df_spark.withColumn(num_borr_col, trim(upper(col(num_borr_col)))) 
    print(f"  Standardized string format for {num_borr_col}.")

    # 3. Create masks for date-dependent logic
    split_date = datetime.date(2018, 3, 31)

    mask_2018q1_prior = col(orig_date_col) <= lit(split_date)
    mask_2018q2_later = col(orig_date_col) > lit(split_date)

    # 4. Conditionally map values with helper function
    mapping_prior = {
        '1': '1_BORROWER',
        '2': '>1_BORROWERS_OLD_RULE' 
    }
    mapping_later = {str(i): str(i) for i in range(1, 11)}

    print(f"  Applying date-dependent mapping for {num_borr_col}...")
    conditional_mapping_expr = when(mask_2018q1_prior, build_categorical_map_exp(col(num_borr_col), mapping_prior, None)) \
                                 .when(mask_2018q2_later, build_categorical_map_exp(col(num_borr_col), mapping_later, None)) \
                                 .otherwise(lit(None).cast(StringType()))
    df_spark = df_spark.withColumn(num_borr_col, conditional_mapping_expr)

    # 5. Create indicator column
    is_missing_col = f"{num_borr_col}_IS_MISSING"
    if is_missing_col not in df_spark.columns:
        df_spark = df_spark.withColumn(is_missing_col, when(col(num_borr_col).isNull(), lit(1)).otherwise(lit(0)).cast(ByteType()))
        print(f"  Created '{is_missing_col}' indicator column.")
        df_spark.select(is_missing_col).groupBy(is_missing_col).count().show(truncate=False)
    else:
        print(f"  '{is_missing_col}' indicator column already exists. Skipping recreation.")

    # 6. Fill remaining NaNs with 'UNKNOWN_NUM_BORROWERS'
    final_nan_count = df_spark.filter(col(num_borr_col).isNull()).count()
    if final_nan_count > 0:
        print(f"  Found {final_nan_count} NaNs (total after all processing). Filling with 'UNKNOWN_NUM_BORROWERS'.")
        df_spark = df_spark.fillna('UNKNOWN_NUM_BORROWERS', subset=[num_borr_col])
    else:
        print(f"  No NaNs found.")

    # 7. Cast to StringType
    df_spark = df_spark.withColumn(num_borr_col, col(num_borr_col).cast(StringType()))
    print(f"  Final datatype of {num_borr_col}: {df_spark.schema[num_borr_col].dataType}.")
    print(f"{num_borr_col} cleaning complete. Remaining NaNs: {df_spark.filter(col(num_borr_col).isNull()).count()}.")
    return df_spark

# Master cleaning function for originations data
def clean_originations_spark(raw_df):
