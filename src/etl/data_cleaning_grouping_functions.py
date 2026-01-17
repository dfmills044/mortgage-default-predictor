from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, substring, length, concat_ws, to_date, sum as spark_sum, count as spark_count, mode as spark_mode, lpad, lit, count, when, median,
    upper, trim
)
from pyspark.sql.types import IntegerType, DateType, StringType, FloatType, ByteType
import pandas as pd

# Function for cleaning columns with the 'Standard Numeric Cleaning Pattern'
def clean_standard_numeric_column_spark(
    df_spark: DataFrame,
    column_name: str,
    null_identifier_value,
    min_clip_value,
    max_clip_value,
    target_spark_dtype
) -> DataFrame:
    if column_name not in df_spark.columns:
        print(f"Column '{column_name}' not found in DataFrame.")
        return df_spark
    
    print(f"Cleaning column '{column_name}'...")

    # 1. Convert to numeric, coercing errors to NaN
    df_spark = df_spark.withColumn(column_name, col(column_name).cast(FloatType()))
    print(f"    Converted column '{column_name}' to FloatType.")

    # 2. Replace null_identifier_value with NaN
    if null_identifier_value is not None:
        initial_identifier_count = df_spark.filter(col(column_name) == lit(null_identifier_value)).count()
        if initial_identifier_count > 0:
            print(f"    Found {initial_identifier_count} instances of '{null_identifier_value}'. Replacing with NaN.")
            df_spark = df_spark.withColumn(column_name, when(col(column_name) == lit(null_identifier_value), lit(None)) \
                                .otherwise(col(column_name)))
        else:
            print(f"    No instances of '{null_identifier_value}' found. Skipping replacement.")
    else:
        print(f"    No null identifier value provided. Skipping replacement.")

    # 3. Create '_IS_MISSING' indicator column
    if f"{column_name}_IS_MISSING" not in df_spark.columns:
        df_spark = df_spark.withColumn(f"{column_name}_IS_MISSING", when(col(column_name).isNull(), 1).otherwise(0).cast(ByteType()))
        print(f"    Created indicator column '{column_name}_IS_MISSING'.")
        df_spark.select(f"{column_name}_IS_MISSING").groupBy(f"{column_name}_IS_MISSING").count().show()
    else:
        print(f"    Indicator column '{column_name}_IS_MISSING' already exists. Skipping creation.")

    # 4. Impute NaNs with the median
    median_val_row = df_spark.agg(median(col(column_name))).first()
    median_val = median_val_row[0] if median_val_row is not None else None

    if median_val is None:
        print(f"    Warning: Median value was Null (all values might be Null). Cannot impute. Leaving NaNs.")
    else:
        initial_nan_count = df_spark.filter(col(column_name).isNull()).count()
        if initial_nan_count > 0:
            print(f"    Found {initial_nan_count} NaNs. Imputing with median: {median_val}")
            df_spark = df_spark.fillna(median_val, subset=[column_name])
        else:
            print(f"    No NaNs found. Skipping imputation.")

    # 5. Clip values to min and max
    print(f"    Clipping {column_name} values to range [{min_clip_value}, {max_clip_value}]")
    df_spark = df_spark.withColumn(column_name, when(col(column_name) < lit(min_clip_value), lit(min_clip_value)) \
                                   .when(col(column_name) > lit(max_clip_value), lit(max_clip_value)) \
                                   .otherwise(col(column_name)))
    current_min_value = df_spark.agg({column_name: "min"}).collect()[0][0] if df_spark.count() > 0 else None
    current_max_value = df_spark.agg({column_name: "max"}).collect()[0][0] if df_spark.count() > 0 else None
    print(f"    Clipped {column_name} values to range [{current_min_value}, {current_max_value}]")

    # 6. Cast to target PySpark DataType
    df_spark = df_spark.withColumn(column_name, col(column_name).cast(target_spark_dtype))
    print(f"    Final datatype of {column_name}: {df_spark.schema[column_name].dataType}")
    print(f"Cleaning of {column_name} complete.")

    return df_spark

# Function for cleaning columns with the 'Binary Flag Cleaning Pattern'
def clean_binary_flag_spark(
    df_spark: DataFrame,
    column_name: str,
    mapping_dict: dict
) -> DataFrame:
    if column_name not in df_spark.columns:
        print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping cleaning.")
        return df_spark
    
    print(f"Cleaning {column_name} columns...")

    # 1. Standardize string format (strip whitespace, make uppercase)
    df_spark = df_spark.withColumn(column_name, upper(trim(col(column_name).cast(StringType()))))
    print(f"    Standardized string format for {column_name}.")

    # 2. Map values based on mapping_dict. Unmapped values become null.
    case_stmt = None
    for key, value in mapping_dict.items():
        lit_value_as_string = lit(value).cast(StringType())
        if case_stmt is None:
            case_stmt = when(col(column_name) == lit(key), lit(value)).when(col(column_name) == lit_value_as_string, lit(value))
        else:
            case_stmt = case_stmt.when(col(column_name) == lit(key), lit(value)).when(col(column_name) == lit_value_as_string, lit(value))
    df_spark = df_spark.withColumn(column_name, case_stmt.otherwise(None))
    print(f"    Mapped values using provided dictionary. Other values are now null.")

    # 3. Fill NaN values with '0'.
    initial_nan_count = df_spark.filter(col(column_name).isNull()).count()
    if initial_nan_count > 0:
        print(f"    Found {initial_nan_count} NaNs. Filling with '0'.")
        df_spark = df_spark.fillna(0, subset=[column_name])
    else:
        print(f"    No NaNs found. Skipping imputation.")

    # 4. Cast to ByteType (int8)
    df_spark = df_spark.withColumn(column_name, col(column_name).cast(ByteType()))
    print(f"    Final datatype of {column_name}: {df_spark.schema[column_name].dataType}")
    print(f"Cleaning of {column_name} complete.")

    return df_spark

# Function for cleaning columns with the 'Standard Categorical Cleaning Pattern'
def clean_standard_categorical_column_spark(
    df_spark: DataFrame,
    column_name: str,
    valid_mapping: dict,
    unknown_value_fill: str,
    special_indicator_configs: dict = None
) -> DataFrame:
    if column_name not in df_spark.columns:
        print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping cleaning.")
        return df_spark
    
    print(f"Cleaning {column_name} columns...")

    # 1. Standardize string format (strip whitespace, make uppercase)
    df_spark = df_spark.withColumn(column_name, upper(trim(col(column_name).cast(StringType()))))
    print(f"    Standardized string format for {column_name}.")

    # --- Handle special indicator columns BEFORE mapping values ---
    # 2. Handle special indicator columns
    if special_indicator_configs:
        print(f"    Creating special indicator columns for {column_name}...")
        original_col_for_indicators = col(column_name)
        for suffix, value_to_flag in special_indicator_configs.items():
            if f"{column_name}_{suffix}" not in df_spark.columns:
                df_spark = df_spark.withColumn(
                    f"{column_name}_{suffix}",
                    when(original_col_for_indicators == lit(value_to_flag), lit(1)).otherwise(lit(0).cast(ByteType()))
                )
                print(f"    Created special indicator columns for {column_name}.")
                df_spark.select(f"{column_name}_{suffix}").groupBy(f"{column_name}_{suffix}").count().show(truncate=False)
            else:
                print(f"    Special indicator column {column_name}_{suffix} already exist. Skipping creation.")
    
    # 3. Map values based on valid_mapping. Unmapped values become null.
    case_stmt = None
    for key, value in valid_mapping.items():
        if case_stmt is None:
            case_stmt = when(col(column_name) == lit(key), lit(value)).when(col(column_name) == lit(value), lit(value))
        else:
            case_stmt = case_stmt.when(col(column_name) == lit(key), lit(value)).when(col(column_name) == lit(value), lit(value))
    df_spark = df_spark.withColumn(column_name, case_stmt.otherwise(None))
    print(f"    Mapped values using provided dictionary. Other values are now null.")

    # 4. Fill NaN values with 'unknown_value_fill'.
    initial_nan_count = df_spark.filter(col(column_name).isNull()).count()
    if initial_nan_count > 0:
        print(f"    Found {initial_nan_count} NaNs. Filling with '{unknown_value_fill}'.")
        df_spark = df_spark.fillna(unknown_value_fill, subset=[column_name])
    else:
        print(f"    No NaNs found. Skipping imputation.")

    # 5. Cast to StringType (category)
    df_spark = df_spark.withColumn(column_name, col(column_name).cast(StringType()))
    print(f"    Final datatype of {column_name}: {df_spark.schema[column_name].dataType}")
    print(f"Cleaning of {column_name} complete.")
    return df_spark

# Function for cleaning columns with the 'Date-Dependent LTV Ratio Cleaning Pattern'
def clean_date_dependent_ltv_ratio_spark(
    df_spark: DataFrame,
    column_name: str,
    orig_date_col: str,
    null_identifier_value,
    clip_bounds_2018q2_prior: tuple,
    clip_bounds_2018q2_post: tuple,
    target_dtype_spark
) -> DataFrame:
    if column_name not in df_spark.columns:
        print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping cleaning.")
        return df_spark
    
    print(f"Cleaning {column_name} columns...")

    # 1. Convert to numeric, coercing errors to NaN
    df_spark = df_spark.withColumn(column_name, col(column_name).cast(FloatType()))
    print(f"    Converted {column_name} to numeric.")

    # 2. Convert null_identifier values to NaN
    initial_identifier_count = df_spark.filter(col(column_name) == lit(null_identifier_value)).count()
    if initial_identifier_count > 0:
        print(f"    Found {initial_identifier_count} null_identifier values. Converting to NaN.")
        df_spark = df_spark.withColumn(column_name, when(col(column_name) == lit(null_identifier_value), lit(None)).otherwise(col(column_name)))
    else:
        print(f"    No null_identifier values found. Skipping conversion.")

    # 3. Create '_IS_MISSING' identifer column
    is_missing_col_name = f"{column_name}_IS_MISSING"
    if is_missing_col_name not in df_spark.columns:
        df_spark = df_spark.withColumn(is_missing_col_name, when(col(column_name).isNull(), lit(1)).otherwise(lit(0)))
        print(f"    Created {is_missing_col_name} column.")
    else:
        print(f"    {is_missing_col_name} column already exists. Skipping creation.")

    # 4. Impute NaNs with median
    median_value_row = df_spark.agg(median(col(column_name))).first()
    median_value = median_value_row[0] if median_value_row is not None else None

    if median_value is None:
        print(f"    Median value for {column_name} is None. Skipping imputation.")
    else:
        initial_nan_count = df_spark.filter(col(column_name).isNull()).count()
        if initial_nan_count > 0:
            print(f"    Found {initial_nan_count} NaNs. Imputing with median value ({median_value}).")
            df_spark = df_spark.fillna(median_value, subset=[column_name])
        else:
            print(f"    No NaNs found. Skipping imputation.")

    # 5. Conditionally clip values based on origination date
    split_date = pd.to_datetime('2018-03-31').date()

    min_prior, max_prior = clip_bounds_2018q2_prior
    min_post, max_post = clip_bounds_2018q2_post

    print(f"    Clipping values based on origination date for {column_name}...")
    df_spark = df_spark.withColumn(column_name,
                                   when(col(orig_date_col) <= lit(split_date), \
                                        when(col(column_name) < lit(min_prior), lit(min_prior)) \
                                        .when(col(column_name) > lit(max_prior), lit(max_prior)) \
                                        .otherwise(col(column_name)))
                                   .when(col(orig_date_col) > lit(split_date), \
                                        when(col(column_name) < lit(min_post), lit(min_post)) \
                                        .when(col(column_name) > lit(max_post), lit(max_post)) \
                                        .otherwise(col(column_name)))
                                   .otherwise(col(column_name)))
    print(f"    Clipping complete.")
    
    df_spark = df_spark.withColumn(column_name, col(column_name).cast(target_dtype_spark))
    print(f"    Final datatype of {column_name}: {df_spark.schema[column_name].dataType}")
    print(f"Cleaning complete for {column_name}.")
    return df_spark

# Function for cleaning columns with the 'Standard Datetime Cleaning Pattern'
def clean_standard_datetime_column_spark(
    df_spark: DataFrame,
    column_name: str,
    date_format_str: str
) -> DataFrame:
    if column_name not in df_spark.columns:
        print(f"Warning: {column_name} not found in DataFrame. Skipping cleaning.")
        return df_spark
    
    print(f"Cleaning {column_name}...")

    current_col_type = df_spark.schema[column_name].dataType

    # 1. Convert to DateTime
    if not isinstance(current_col_type, DateType):
        df_spark = df_spark.withColumn(column_name, to_date(col(column_name).cast(StringType()), date_format_str))
        print(f"    Converted {column_name} to date.")
    else:
        print(f"    {column_name} is already a date. Skipping conversion.")
        
    # 2. Create '_IS_MISSING' indicator column
    is_missing_col_name =f"{column_name}_IS_MISSING"
    if is_missing_col_name not in df_spark.columns:
        df_spark = df_spark.withColumn(is_missing_col_name, when(col(column_name).isNull(), 1).otherwise(0).cast(ByteType()))
        print(f"    Created {is_missing_col_name} column.")
        df_spark.select(is_missing_col_name).groupBy(is_missing_col_name).count().show(truncate=False)
    else:
        print(f"    {is_missing_col_name} column already exists. Skipping creation.")
    
    # 3. Impute NaT values with the mode
    initial_null_count = df_spark.filter(col(column_name).isNull()).count()
    if initial_null_count > 0:
        print(f"    Found {initial_null_count} NaT values in {column_name}. Dropping these rows.")
        df_spark = df_spark.dropna(subset=[column_name])
    else:
        print(f"    No NaT values found in {column_name}. No rows dropped.")
    print(f"    Final datatype of {column_name}: {df_spark.schema[column_name].dataType}")
    print(f"Cleaning complete for {column_name}.")
    return df_spark

# Function for cleaning columns with the 'Variable String Categorical Cleaning Pattern'
def clean_variable_string_categorical_column_spark(
    df_spark: DataFrame,
    column_name: str,
    unknown_fill_value: str
) -> DataFrame:
    if column_name not in df_spark.columns:
        print(f"Warning: {column_name} not found in DataFrame. Skipping cleaning.")
        return df_spark
    
    print(f"Cleaning {column_name}...")

    # 1. Standardize string format
    df_spark = df_spark.withColumn(column_name, col(column_name).cast(StringType()))
    df_spark = df_spark.withColumn(column_name, upper(trim(col(column_name))))
    print(f"    Standardized {column_name} to uppercase string.")
    
    # 2. Create '_IS_MISSING' indicator column
    missing_placeholders = {'NAN', ''}
    mask_is_missing_original = col(column_name).isin(missing_placeholders)

    is_missing_col_name =f"{column_name}_IS_MISSING"
    if is_missing_col_name not in df_spark.columns:
        df_spark = df_spark.withColumn(is_missing_col_name, when(mask_is_missing_original, 1).otherwise(0).cast(ByteType()))
        print(f"    Created {is_missing_col_name} column.")
        df_spark.select(is_missing_col_name).groupBy(is_missing_col_name).count().show(truncate=False)
    else:
        print(f"    {is_missing_col_name} column already exists. Skipping creation.")
    
    # 3. Impute missing values with the unknown_fill_value
    initial_invalid_count = df_spark.filter(mask_is_missing_original).count()
    if initial_invalid_count > 0:
        print(f"    Found {initial_invalid_count} invalid values. Imputing with {unknown_fill_value}.")
        df_spark = df_spark.withColumn(column_name, \
                                            when(col(column_name) == lit(unknown_fill_value), lit(unknown_fill_value)) \
                                            .when(mask_is_missing_original, lit(unknown_fill_value)) \
                                            .otherwise(col(column_name)))
    else:
        print(f"    No invalid values found. Skipping imputation.")

    # 4. Cast to StringType
    df_spark = df_spark.withColumn(column_name, col(column_name).cast(StringType()))
    print(f"    Final datatype of {column_name}: {df_spark.schema[column_name].dataType}")
    print(f"Cleaning complete for {column_name}.")
    return df_spark

# Function for cleaning columns with the 'Financial Cost Cleaning Pattern'
def clean_financial_cost_column_spark(
    df_spark: DataFrame,
    col_name: str,
    indicator_col_name: str
):
    if col_name not in df_spark.columns:
        print(f"Warning: {col_name} column not found. Skipping.")
        return df_spark
    
    print(f"Cleaning financial cost column: {col_name}...")

    # Cast to float and force negative and zero numbers to NaN
    # Treat everything except positive values as missing for the indicator column
    df_spark = df_spark.withColumn(
        col_name,
        when(col(col_name).cast(FloatType()) <= 0, lit(None),)
        .otherwise(col(col_name).cast(FloatType()))
    )
    print(f"    Cast {col_name} to float32 and force negative and zero numbers to NaN.")

    # Create indicator column (1 if positive cost is present, 0 if NULL/Negative/Zero)
    df_spark = df_spark.withColumn(
        indicator_col_name,
        when(col(col_name).isNotNull(), lit(1)).otherwise(lit(0)).cast(ByteType())
    )
    print(f"    Created indicator column: {indicator_col_name}")

    # Fill NaN values with 0
    df_spark = df_spark.fillna(0, subset=[col_name])
    print("    Filled NaN values with 0.")

    # Cast to float32 for memory efficiency
    df_spark = df_spark.withColumn(col_name, col(col_name).cast(FloatType()))

    print(f"    Final datatype of {col_name}: {df_spark.schema[col_name].dataType}")
    print(f"{col_name} cleaning complete.")

    return df_spark