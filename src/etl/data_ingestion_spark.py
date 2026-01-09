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
