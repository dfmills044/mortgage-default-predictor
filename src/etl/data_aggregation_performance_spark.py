from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import (col, sum as spark_sum, count as spark_count, mode as lit, lag, coalesce, when)
from pyspark.sql.window import Window
from pyspark.sql.types import ByteType, IntegerType
from aggregation_functions import agg_expressions

window_spec = Window.partitionBy('LOAN_SEQUENCE_NUMBER').orderBy('MONTHLY_REPORTING_PERIOD')

# Function for dropping bad rows
def drop_bad_loan_ids(df_spark: DataFrame) -> DataFrame:
    """
    Bad Rows are defined as one of the following four scenarios:
    1. Rows with duplicate LOAN_SEQUENCE_NUMBER and MONTHLY_REPORTING_PERIOD values. In this case we keep the first one.
    2. Rows for loans that continue after the loan was terminated. In this case we clip the rows that appear after the termination date for the loan.
    3. Loans that have unusual delinquency jumps (e.g., the monthly reporting period increases by one month, but delinquency status increases by more tha 1). 
       In this case we drop all rows related to that LOAN_SEQUENCE_NUMBER.
    4. Loan that have unexplained jumps in CURRENT_ACTUAL_UPB. 
       If the jump cannot be explained by a modification, payment deferral, increase in non-interest bearing UPB, borrower assistance program, or disaster event, we drop all rows related to that loan number.
    """
    # 1. Deduplication - drop rows where LOAN_SEQUENCE_NUMBER and MONTHLY_REPORTING_PERIOD are the same (keeps the first one it finds)
    print(f"Total rows before dropping bad rows: {df_spark.count()}")
    df_spark = df_spark.dropDuplicates(['LOAN_SEQUENCE_NUMBER', 'MONTHLY_REPORTING_PERIOD'])

    # 2a. Find the earliest month a loan was terminated
    termination_dates = df_spark.filter(col('ZERO_BALANCE_CODE') != 'NO_ZERO_BALANCE_CODE').groupBy('LOAN_SEQUENCE_NUMBER') \
        .agg(F.min('MONTHLY_REPORTING_PERIOD').alias('termination_month'))

    # 2b. Join and filter outh records that occur after the termination month
    df_spark = df_spark.join(termination_dates, ['LOAN_SEQUENCE_NUMBER'], 'left')
    df_spark = df_spark.filter((col('termination_month').isNull()) | (col('MONTHLY_REPORTING_PERIOD') <= col('termination_month'))) \
        .drop('termination_month')

    # 3. Identify Loan IDs with delinquency jumps
    bad_delinq_ids = df_spark.withColumn("prev_status", lag("CURRENT_DELINQUENCY_STATUS").over(window_spec)) \
        .filter(col("CURRENT_DELINQUENCY_STATUS") - col("prev_status") > 1) \
            .select("LOAN_SEQUENCE_NUMBER").distinct()
    
    # 4. Identify Loan IDs with unexplained UPB increases
    df_spark = df_spark.withColumn("prev_nib", lag("CURRENT_NON_INTEREST_BEARING_UPB").over(window_spec))
    is_modified = (col('MOD_FLAG') != 'NOT_MODIFIED')
    is_deferral = (col('PAYMENT_DEFERRAL_FLAG') != 'NOT_PAYMENT_DEFERRAL')
    is_step_mod = (col('STEP_MOD_FLAG') != 'LOAN_NOT_MODIFIED')
    is_assisted = (col('BORROWER_ASSISTANCE_STATUS_CODE') != 'NO_BORR_ASSIST_CODE')
    is_disaster = (col('DELINQUENCY_DUE_TO_DISASTER') == 1)
    nib_jump = (col('CURRENT_NON_INTEREST_BEARING_UPB') > coalesce(col('prev_nib'), lit(0)) + 100)
    bad_upb_ids = df_spark.withColumn("prev_upb", lag("CURRENT_ACTUAL_UPB").over(window_spec)) \
        .filter((col("CURRENT_ACTUAL_UPB") > col("prev_upb") + 1000) & (~is_modified) & (~is_deferral) & (~is_step_mod) & (~is_assisted) & (~is_disaster) & (~nib_jump)) \
            .select("LOAN_SEQUENCE_NUMBER").distinct()

    # 5. Identify loans with UPB missing at any point
    bad_missing_upb_ids = df_spark.filter(
        (col('CURRENT_ACTUAL_UPB_IS_MISSING') == 1) &
        (col('ZERO_BALANCE_CODE') == 'NO_ZERO_BALANCE_CODE')
    ).select('LOAN_SEQUENCE_NUMBER').distinct()

    # 6. Combine and Filter
    excluded_loan_ids = bad_delinq_ids.union(bad_upb_ids).union(bad_missing_upb_ids).distinct()
    print(f"Total loans to be excluded: {excluded_loan_ids.count()}")

    # Apply filter to dataframe
    df_spark = df_spark.join(excluded_loan_ids, ['LOAN_SEQUENCE_NUMBER'], 'left_anti')
    df_spark = df_spark.drop("prev_nib")
    print(f"Total rows after dropping bad rows: {df_spark.count()}")
    return df_spark

# Function for creating the IS_IN_DEFAULT column
def create_default_col(df_spark: DataFrame) -> DataFrame:
    """
    This function creates a new column IS_IN_DEFAULT that indicates if a loan is in default at a certain point in time.
    A loan is considered to be in default if any of the following statements are true:
    1. The loan is in delinquency for 90 or more days (CURRENT_DELINQUENCY_STATUS >= 3)
    2. The loan has a delinquency status of RA (CURRENT_DELINQUENCY_STATUS_IS_RA == 1)
    3. The loan has a zero balance code of a credit event (listed in the bad_exit_codes array)
    4. The loan's DDLPI is 90 or more days in the past
    """
    # Define the bad exit codes that indicate default in ZERO_BALANCE_CODE column
    bad_exit_codes = ['THIRD_PARTY_SALE', 'SHORT_SALE_OR_CHARGE_OFF', 'REO_DISPOSITION', 'WHOLE_LOAN_SALE', 'REPREFORMING_LOAN_SECURITIZATION']

    # Definition of IS_IN_DEFAULT (row level)
    # 1. Delinquency is 90+ days (3 or more) or is 'RA'
    # 2. OR Zero Balance Code is a Credit Event (is in bad_exit_codes)
    # 3. OR DDLPI is 90 days or more in the past
    df_spark = df_spark.withColumn(
        'IS_IN_DEFAULT',
        when(
            (col('CURRENT_DELINQUENCY_STATUS') >= 3) |
            (col('CURRENT_DELINQUENCY_STATUS_IS_RA') == 1) |
            (col('ZERO_BALANCE_CODE').isin(bad_exit_codes)) |
            (F.months_between(col('MONTHLY_REPORTING_PERIOD'), col('DDLPI')) >= 3), lit(1)
        ).otherwise(lit(0)).cast(ByteType())
    )
    return df_spark

# Function for mapping exit categories
def get_exit_category_logic(col_name):
    return when(col(col_name).isin(['THIRD_PARTY_SALE', 'SHORT_SALE_OR_CHARGE_OFF', 'REO_DISPOSITION', 'WHOLE_LOAN_SALE']), 'Default') \
        .when(col(col_name) == 'PREPAID_OR_MATURED', 'Prepaid') \
            .when(col(col_name).isin(['REPERFORMING_LOAN_SECURITIZATION', 'DEFECT_PRIOR_TO_OTHER_EVENT']), 'Repurchased') \
                .otherwise('Active')

# Function for creating column tracking loan age (independent of loan modifications)
def create_chrono_age_col(df_spark: DataFrame) -> DataFrame:
    # Identify observation start date for each loan
    start_dates = df_spark.groupBy('LOAN_SEQUENCE_NUMBER').agg(F.min('MONTHLY_REPORTING_PERIOD').alias('START_DATE'))

    # Join back and calculate a reset-proof age
    return df_spark.join(start_dates, on='LOAN_SEQUENCE_NUMBER') \
        .withColumn('CHRONO_AGE', F.round(F.months_between(col('MONTHLY_REPORTING_PERIOD'), col('START_DATE'))).cast(IntegerType()))

# Function for creating default labels (necessary as these columns depend on the entire loan history, not just pre-default)
def create_default_labels(df_spark: DataFrame) -> DataFrame:
    # Determine default labels for each loan, along with zero balance aggregations and delinquent interest aggregations
    # After aggregation, combine this with the aggregated features
    # df_agg_features.join(labels_df, on="LOAN_SEQUENCE_NUMBER", how="inner")
    return df_spark.groupBy('LOAN_SEQUENCE_NUMBER').agg(
        F.max('IS_IN_DEFAULT').alias('LABEL_DEFAULT'),
        F.sum('IS_IN_DEFAULT').alias('TOTAL_MONTHS_IN_DEFAULT'),
        F.min(
            when(col('IS_IN_DEFAULT') == 1, col('CHRONO_AGE'))
        ).alias('AGE_AT_FIRST_DEFAULT'),
        F.last('IS_IN_DEFAULT').alias('IS_TERMINAL_DEFAULT'),
        F.last('ZERO_BALANCE_CODE').alias('ACTUAL_TERMINAL_ZBC'),
        F.last(get_exit_category_logic('ZERO_BALANCE_CODE')).alias('ACTUAL_EXIT_CATEGORY'),
        F.max('DELINQUENT_ACCRUED_INTEREST').alias('TOTAL_ACTUAL_LOSS_AMOUNT'),
        F.max('HAS_DELINQUENT_INTEREST').alias('WAS_CREDIT_LOSS_EVENT'),
        F.last('DELINQUENCY_DUE_TO_DISASTER').alias('FINAL_EXIT_WAS_DISASTER_RELATED')
    )

# Function for creating mask of data before default (necessary as the model should be trained only on pre-default data)
def create_pre_default_mask(df_spark: DataFrame) -> DataFrame:
    # Find first month each loan defaulted
    first_default_df = df_spark.filter(col('IS_IN_DEFAULT') == 1) \
        .groupBy('LOAN_SEQUENCE_NUMBER') \
            .agg(F.min('MONTHLY_REPORTING_PERIOD').alias('FIRST_DEFAULT'))

    # Join back to main dataframe
    df_with_cutoff = df_spark.join(first_default_df, on='LOAN_SEQUENCE_NUMBER', how='left')

    # Create mask: Keep rows where reporting period < first default month
    # If loan never defaulted (FIRST_DEFAULT is null), keep all rows
    # Apply aggregation to this mask, not the original cleaned df
    return df_with_cutoff.filter(
        (col('FIRST_DEFAULT').isNull()) |
        (col('MONTHLY_REPORTING_PERIOD') < col('FIRST_DEFAULT'))
    )

# Function for aggregating performance data (given bad rows have been dropped)
def aggregate_and_join(
    pre_default_mask: DataFrame,
    default_labels: DataFrame,
    agg_functions
):
    # Perform heavy lifting (aggregation)
    df_agg = pre_default_mask.groupBy('LOAN_SEQUENCE_NUMBER').agg(*agg_functions)

    # Grand Join (Not originations yet)
    return df_agg.join(default_labels, on='LOAN_SEQUENCE_NUMBER', how='inner')