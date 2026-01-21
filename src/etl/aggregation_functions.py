from pyspark.sql import functions as F
from pyspark.sql.functions import (when, col, lit)

# Set CURRENT_DELINQUENCY_STATUS fow rows with CURRENT_DELINQUENCY_STATUS_IS_RA = 1 to 18
delinq_clean = when(col('CURRENT_DELINQUENCY_STATUS_IS_RA') == 1, lit(18)).otherwise(col('CURRENT_DELINQUENCY_STATUS'))

# Array of aggregation expressions to be used during aggregation
agg_expressions = [
    # MONTHLY_REPORTING_PERIOD
    F.max('MONTHLY_REPORTING_PERIOD').alias('LAST_MONTHLY_REPORTING_PERIOD'),
    F.min('MONTHLY_REPORTING_PERIOD').alias('FIRST_MONTHLY_REPORTING_PERIOD'),
    F.count('MONTHLY_REPORTING_PERIOD').alias('PERFORMANCE_MONTHS_OBSERVED'),
    # CURRENT_ACTUAL_UPB
    F.first('CURRENT_ACTUAL_UPB').alias('INITIAL_ACTUAL_UPB'),
    F.last(
        when(col('CURRENT_ACTUAL_UPB') > 0, col('CURRENT_ACTUAL_UPB')),
        ignorenulls=True
    ).alias('FINAL_NON_ZERO_UPB'),
    F.max('CURRENT_ACTUAL_UPB').alias('MAX_ACTUAL_UPB'),
    F.last('CURRENT_ACTUAL_UPB').alias('LAST_ACTUAL_UPB'),
    # CURRENT_DELINQUENCY_STATUS
    F.max(delinq_clean).alias('MAX_DELINQUENCY_STATUS'),
    F.sum(when(delinq_clean > 0, 1).otherwise(0)).alias('NUMBER_OF_LATE_MONTHS'),
    F.last(delinq_clean).alias('TERMINAL_DELINQ_STATUS'),
    F.coalesce(F.stddev(delinq_clean), lit(0)).alias('DELINQUENCY_STATUS_STDDEV'),
    F.max('CURRENT_DELINQUENCY_STATUS_IS_RA').alias('EVER_RA'),
    F.max('CURRENT_DELINQUENCY_STATUS_IS_MISSING').alias('DELINQUENCY_STATUS_EVER_MISSING'),
    # LOAN_AGE
    F.first('LOAN_AGE').alias('INITIAL_LOAN_AGE'),
    F.last('LOAN_AGE').alias('FINAL_LOAN_AGE'),
    F.max('LOAN_AGE').alias('MAX_LOAN_AGE'),
    F.min('LOAN_AGE').alias('MIN_LOAN_AGE'),
    F.max('LOAN_AGE_IS_MISSING').alias('LOAN_AGE_EVER_MISSING'),
    # MOD_FLAG
    F.max(
        when(col('MOD_FLAG') != 'NOT_MODIFIED', 1).otherwise(0)
    ).alias('EVER_MODIFIED'),
    F.sum(
        when(col('MOD_FLAG') == 'CURRENT_PERIOD_MOD', 1).otherwise(0)
    ).alias('NUMBER_OF_MODIFICATIONS'),
    # ZERO_BALANCE_CODE
    F.last('ZERO_BALANCE_CODE').alias('PRE_DEFAULT_ZBC'),
    F.when(F.last('ZERO_BALANCE_CODE') == 'NO_ZERO_BALANCE_CODE', 1).otherwise(0).alias('IS_ACTIVE_AT_SNAPSHOT'),
    # ZERO_BALANCE_EFFECT_DATE
    F.last('ZERO_BALANCE_EFFECT_DATE', ignorenulls=True).alias('LAST_ZBE_DATE'),
    # CURRENT_INTEREST_RATE
    F.first('CURRENT_INTEREST_RATE').alias('INITIAL_INTEREST_RATE'),
    F.last('CURRENT_INTEREST_RATE').alias('LAST_INTEREST_RATE'),
    F.max('CURRENT_INTEREST_RATE').alias('MAX_INTEREST_RATE'),
    F.min('CURRENT_INTEREST_RATE').alias('MIN_INTEREST_RATE'),
    F.coalesce(F.stddev('CURRENT_INTEREST_RATE'), lit(0)).alias('INTEREST_RATE_STDDEV'),
    F.max('CURRENT_INTEREST_RATE_IS_MISSING').alias('INTEREST_RATE_EVER_MISSING'),
    # CURRENT_NON_INTEREST_BEARING_UPB
    F.max(
        when(col('CURRENT_NON_INTEREST_BEARING_UPB') > 0, 1).otherwise(0)
    ).alias('EVER_HAD_NIB_UPB'),
    F.max('CURRENT_NON_INTEREST_BEARING_UPB').alias('MAX_NIB_UPB'),
    F.last('CURRENT_NON_INTEREST_BEARING_UPB').alias('LAST_NIB_UPB'),
    # DDLPI
    F.last('DDLPI', ignorenulls=True).alias('FINAL_DDLPI_DATE'),
    F.last(
        F.months_between(col('MONTHLY_REPORTING_PERIOD'), col('DDLPI')), ignorenulls=True
    ).alias('FINAL_PAYMENT_GAP_MONTHS'),
    F.max(
        F.months_between(col('MONTHLY_REPORTING_PERIOD'), col('DDLPI'))
    ).alias('MAX_PAYMENT_GAP_MONTHS'),
    # CUMULATIVE_MOD_COST
    F.max('CUMULATIVE_MOD_COST').alias('MAX_CUMULATIVE_MOD_COST'),
    F.max('CUMULATIVE_MOD_COST_IS_MODIFIED').alias('EVER_HAD_MOD_COST'),
    # STEP_MOD_FLAG
    F.max(
        when(col('STEP_MOD_FLAG') == 'STEP_MOD', 1).otherwise(0)
    ).alias('EVER_STEP_MOD'),
    F.max(
        when(col('STEP_MOD_FLAG') == 'NON_STEP_MOD', 1).otherwise(0)
    ).alias('EVER_FIXED_MOD'),
    F.last(
        when(col('STEP_MOD_FLAG') == 'STEP_MOD', 1).otherwise(0)
    ).alias('IS_CURRENTLY_STEP_MOD'),
    # PAYMENT_DEFERRAL_FLAG
    F.max(
        when(col('PAYMENT_DEFERRAL_FLAG') != 'NOT_PAYMENT_DEFERRAL', 1).otherwise(0)
    ).alias('EVER_PAYMENT_DEFERRAL'),
    F.last(
        when(col('PAYMENT_DEFERRAL_FLAG') == 'CURRENT_PERIOD', 1).otherwise(0)
    ).alias('IS_RECENTLY_DEFERRED'),
    F.sum(
        when(col('PAYMENT_DEFERRAL_FLAG') == 'CURRENT_PERIOD', 1).otherwise(0)
    ).alias('DEFERRAL_EVENT_COUNT'),
    # ELTV
    F.last('ELTV', ignorenulls=True).alias('LAST_ELTV'),
    F.max('ELTV').alias('MAX_ELTV'),
    F.min('ELTV').alias('MIN_ELTV'),
    F.max(when(col('ELTV') > 100, 1).otherwise(0)).alias('EVER_UNDERWATER'),
    F.first('ELTV').alias('INITIAL_ELTV'),
    F.last('ELTV_IS_MISSING').alias('LAST_ELTV_IS_MISSING'),
    F.first('ELTV_IS_MISSING').alias('INITIAL_ELTV_IS_MISSING'),
    F.max('ELTV_IS_MISSING').alias('ELTV_EVER_MISSING'),
    # DELINQUENT_ACCRUED_INTEREST
    F.max('DELINQUENT_ACCRUED_INTEREST').alias('PRE_DEFAULT_MAX_ACCRUED_INTEREST'),
    F.max('HAS_DELINQUENT_INTEREST').alias('PRE_DEFAULT_EVER_HAD_LOSS_SIGNS'),
    # DELINQUENCY_DUE_TO_DISASTER
    F.max('DELINQUENCY_DUE_TO_DISASTER').alias('PRE_DEFAULT_EVER_DISASTER_AFFECTED'),
    F.last('DELINQUENCY_DUE_TO_DISASTER').alias('PRE_DEFAULT_CURRENTLY_DISASTER_AFFECTED'),
    F.max('DISASTER_FLAG_MISSING').alias('IS_PRE_2014_VINTAGE_REPORTING'),
    # BORROWER_ASSISTANCE_STATUS_CODE
    F.max(
        when(col('BORROWER_ASSISTANCE_STATUS_CODE') == 'FORBEARANCE', 1).otherwise(0)
    ).alias('EVER_FORBEARANCE'),
    F.max(
        when(col('BORROWER_ASSISTANCE_STATUS_CODE') == 'REPAYMENT', 1).otherwise(0)
    ).alias('EVER_REPAYMENT'),
    F.max(
        when(col('BORROWER_ASSISTANCE_STATUS_CODE') == 'TRIAL_PERIOD', 1).otherwise(0)
    ).alias('EVER_TRIAL_PERIOD'),
    F.last('BORROWER_ASSISTANCE_STATUS_CODE').alias('LAST_ASSISTANCE_STATUS'),
    # CURRENT_MONTH_MOD_COST
    F.avg('CURRENT_MONTH_MOD_COST').alias('AVG_MONTHLY_MOD_COST'),
    F.max('CURRENT_MONTH_MOD_COST').alias('MAX_MONTHLY_MOD_COST'),
    F.last('CURRENT_MONTH_MOD_COST').alias('LAST_MONTHLY_MOD_COST'),
    F.max('CURRENT_MONTH_IS_MODIFIED').alias('EVER_MONTHLY_MODIFIED')
]