from pyspark.sql import DataFrame
from pyspark.sql.functions import col

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