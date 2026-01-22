from pyspark.sql import DataFrame
from data_cleaning_originations_spark import clean_originations_spark
from data_cleaning_performance_spark import clean_performance_spark
from data_aggregation_performance_spark import aggregate_performance
from data_ingestion_spark import (load_origination_data_spark, load_performance_data_spark)

# Master cleaning, aggregation, and merging pipeline function
def clean_agg_merge_freddie_mac_data(
    spark, # Pass in the active SparkSession into the master pipeline function
    perf_data_file_path: str,
    orig_data_file_path: str
    ) -> DataFrame:
    
    # Load originations and performance datasets
    orig_df_raw = load_origination_data_spark(spark, orig_data_file_path)
    perf_df_raw = load_performance_data_spark(spark, perf_data_file_path)

    # Clean originations data
    orig_df_cleaned = clean_originations_spark(orig_df_raw)

    # Clean performance data
    perf_df_cleaned = clean_performance_spark(perf_df_raw)

    # Aggregate performance data
    perf_df_agg = aggregate_performance(perf_df_cleaned)

    # Return the joined performance and originations dataframe
    return perf_df_agg.join(orig_df_cleaned, on='LOAN_SEQUENCE_NUMBER', how='inner')