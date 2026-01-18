from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import (col, sum as spark_sum, count as spark_count, mode as lit, lag, coalesce, when)
from pyspark.sql.window import Window
from pyspark.sql.types import ByteType

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

    # 5. Combine and Filter
    excluded_loan_ids = bad_delinq_ids.union(bad_upb_ids).distinct()
    print(f"Total loans to be excluded: {excluded_loan_ids.count()}")

    # Apply filter to dataframe
    df_spark = df_spark.join(excluded_loan_ids, ['LOAN_SEQUENCE_NUMBER'], 'left_anti')
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