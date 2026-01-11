from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType

# Function for loading originations data
def load_origination_data_spark(spark_session, file_path):
    print(f"Loading origination data from {file_path}")

    # Define schema
    orig_data_schema = StructType([
        StructField('CREDIT_SCORE', StringType(), True),
        StructField('FIRST_PAYMENT_DATE', StringType(), True),
        StructField('FIRST_TIME_HOMEBUYER_FLAG', StringType(), True),
        StructField('MATURITY_DATE', StringType(), True),
        StructField('MSA_OR_MET_DIV', StringType(), True),
        StructField('MI_PERCENT', StringType(), True),
        StructField('NUMBER_OF_UNITS', StringType(), True),
        StructField('OCCUPANCY_STATUS', StringType(), True),
        StructField('ORIGINAL_CLTV', StringType(), True),
        StructField('ORIGINAL_DTI_RATIO', StringType(), True),
        StructField('ORIGINAL_UPB', StringType(), True),
        StructField('ORIGINAL_LTV', StringType(), True),
        StructField('ORIGINAL_INTEREST_RATE', StringType(), True),
        StructField('CHANNEL', StringType(), True),
        StructField('PPM_FLAG', StringType(), True),
        StructField('AMORTIZATION_TYPE', StringType(), True),
        StructField('PROPERTY_STATE', StringType(), True),
        StructField('PROPERTY_TYPE', StringType(), True),
        StructField('POSTAL_CODE', StringType(), True),
        StructField('LOAN_SEQUENCE_NUMBER', StringType(), True),
        StructField('LOAN_PURPOSE', StringType(), True),
        StructField('ORIGINAL_LOAN_TERM', StringType(), True),
        StructField('NUMBER_OF_BORROWERS', StringType(), True),
        StructField('SELLER_NAME', StringType(), True),
        StructField('SERVICER_NAME', StringType(), True),
        StructField('SUPER_CONFORMING_FLAG', StringType(), True),
        StructField('PRE_RELIEF_LOAN_NUMBER', StringType(), True),
        StructField('PROGRAM_INDICATOR', StringType(), True),
        StructField('RELIEF_REFI_INDICATOR', StringType(), True),
        StructField('PROP_VALUATION_METHOD', StringType(), True),
        StructField('IO_INDICATOR', StringType(), True),
        StructField('MI_CANCEL_INDICATOR', StringType(), True)
    ])

    df_spark = spark_session.read.csv(file_path, header=False, sep="|", schema=orig_data_schema)

    print(f"Loaded {df_spark.count()} records with {len(df_spark.columns)} columns from {file_path}")
    df_spark.printSchema()

    return df_spark

def load_performance_data_spark(spark_session, file_path):
    print(f"Loading performance data from {file_path}")

    # Define schema
    performance_data_schema = StructType([
        StructField('LOAN_SEQUENCE_NUMBER', StringType(), True),
        StructField('MONTHLY_REPORTING_PERIOD', StringType(), True),
        StructField('CURRENT_ACTUAL_UPB', StringType(), True),
        StructField('CURRENT_DELINQUENCY_STATUS', StringType(), True),
        StructField('LOAN_AGE', StringType(), True),
        StructField('MONTHS_REMAINING_TO_LEGAL_MATURITY', StringType(), True),
        StructField('DEFECT_SETTLEMENT_DATE', StringType(), True),
        StructField('MOD_FLAG', StringType(), True),
        StructField('ZERO_BALANCE_CODE', StringType(), True),
        StructField('ZERO_BALANCE_EFFECT_DATE', StringType(), True),
        StructField('CURRENT_INTEREST_RATE', StringType(), True),
        StructField('CURRENT_NON_INTEREST_BEARING_UPB', StringType(), True),
        StructField('DDLPI', StringType(), True),
        StructField('MI_RECOVERIES', StringType(), True),
        StructField('NET_SALE_PROCEEDS', StringType(), True),
        StructField('NON_MI_RECOVERIES', StringType(), True),
        StructField('TOTAL_EXPENSES', StringType(), True),
        StructField('LEGAL_COSTS', StringType(), True),
        StructField('MAINTENANCE_AND_PRESERVE_COSTS', StringType(), True),
        StructField('TAXES_AND_INSURANCE', StringType(), True),
        StructField('MISC_EXPENSES', StringType(), True),
        StructField('ACTUAL_LOSS_CALC', StringType(), True),
        StructField('CUMULATIVE_MOD_COST', StringType(), True),
        StructField('INTEREST_RATE_STEP_INDICATOR', StringType(), True),
        StructField('PAYMENT_DEFERRAL_FLAG', StringType(), True),
        StructField('ELTV', StringType(), True),
        StructField('ZERO_BALANCE_REMOVAL_UPB', StringType(), True),
        StructField('DELINQUENT_ACCRUED_INTEREST', StringType(), True),
        StructField('DELINQUENCY_DUE_TO_DISASTER', StringType(), True),
        StructField('BORROWER_ASSISTANCE_STATUS_CODE', StringType(), True),
        StructField('CURRENT_MONTH_MOD_COST', StringType(), True),
        StructField('INTEREST_BEARING_UPB', StringType(), True)
    ])

    df_spark = spark_session.read.csv(file_path, header=False, sep='|', schema=performance_data_schema)
    
    print(f"    Loaded {df_spark.count()} records with {len(df_spark.columns)} columns from {file_path}")
    df_spark.printSchema()

    return df_spark