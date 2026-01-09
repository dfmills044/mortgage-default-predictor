import pandas as pd
import numpy as np

# Function for validating 'LOAN_SEQUENCE_NUMBER' column
def validate_loan_sequence_number_performance(df):
    loan_number = 'LOAN_SEQUENCE_NUMBER'

    if loan_number in df.columns:
        print(f"Validating {loan_number} column...")

        if not pd.api.types.is_string_dtype(df[loan_number]):
            print(f"    Converting {loan_number} to string (object) dtype for consistent validation and joining.")
            df[loan_number] = df[loan_number].astype(str)

        df[loan_number] = df[loan_number].str.strip()

        initial_null_count = df[loan_number].isna().sum()
        if initial_null_count > 0:
            print(f"    Warning: {initial_null_count} null values found in {loan_number}. These rows cannot be uniquely identified or joined.")
            df.dropna(subset=[loan_number], inplace=True)
            print(f"    Dropped rows with null {loan_number}. Remaining rows: {len(df)}")
        else:
            print(f"    No null values found in {loan_number}.")

        print(f"{loan_number} validation complete.")

        return df

    else:
        print(f"{loan_number} column not found. Skipping validation.")
        return df

# Function for cleaning 'MONTHLY_REPORTING_PERIOD' column, creating 'REPORTING_DATE' column, and filtering for between 2023-01-01 and 2024-04-01
def clean_and_filter_performance_dates(df):
    mrp_col = "MONTHLY_REPORTING_PERIOD"
    reporting_date_col = "REPORTING_DATE"

    if mrp_col in df.columns:
        print(f"Cleaning and filtering performance data by {mrp_col}...")

        df[reporting_date_col] = pd.to_datetime(df[mrp_col].astype(str), format="%Y%m", errors="coerce")
        print(f"    Converted {mrp_col} (YYYYMM format) to datetime {reporting_date_col}.")

        initial_nat_count = df[reporting_date_col].isna().sum()
        if initial_nat_count > 0:
            print(f"    Warning: {initial_nat_count} NaT values found in {reporting_date_col}. These rows will be dropped during filtering as they cannot be dated.")
        else:
            print(f"    No NaT values found in {reporting_date_col}.")

        start_filter_date = pd.to_datetime("2023-01-01")
        end_filter_date = pd.to_datetime("2024-04-01")

        rows_before_filter = len(df)

        df = df[(df[reporting_date_col] >= start_filter_date) & (df[reporting_date_col] <= end_filter_date)].copy()
        rows_after_filter = len(df)

        df = df.reset_index(drop=True)

        print(f"    Filtered data to {start_filter_date.strftime('%Y-%m-%d')} to {end_filter_date.strftime('%Y-%m-%d')}.")
        print(f"    Rows before filter: {rows_before_filter}, Rows after filter: {rows_after_filter}. Dropped {rows_before_filter-rows_after_filter} rows.")

        print(f"Cleaning and filtering by {mrp_col} complete.")

        return df
    else:
        print(f"{mrp_col} column not found. Skipping cleaning and filtering.")
        return df

# Function for cleaning 'CURRENT_ACTUAL_UPB' column
def clean_current_upb(df):
    current_upb_col = 'CURRENT_ACTUAL_UPB'

    if current_upb_col in df.columns:
        print(f"Cleaning {current_upb_col} column...")

        df[current_upb_col] = pd.to_numeric(df[current_upb_col], errors='coerce')

        df[f"{current_upb_col}_IS_MISSING"] = df[current_upb_col].isna().astype("int8")
        print(f"    Created '{current_upb_col}_IS_MISSING' indicator column.")
        print(f"    '{current_upb_col}_IS_MISSING' value counts:\n{df[f"{current_upb_col}_IS_MISSING"].value_counts()}")

        median_upb = df[current_upb_col].median()
        print(f"    Calculated median {current_upb_col} value: {median_upb}")

        final_missing_count = df[current_upb_col].isna().sum()
        if final_missing_count > 0:
            print(f"    Found {final_missing_count} NaN values. Imputing with median ({median_upb}).")
            df[current_upb_col] = df[current_upb_col].fillna(median_upb)
        else:
            print(f"    No NaN values found in {current_upb_col}.")

        original_min = df[current_upb_col].min()
        original_max = df[current_upb_col].max()

        if original_min < 0 or original_max > 3000000:
            print(f"    Warning: Found {current_upb_col} values outside of 0-3000000 ({original_min}-{original_max}). Clipping.")
            df[current_upb_col] = np.clip(df[current_upb_col], 0, 3000000)
        else:
            print(f"    All {current_upb_col} in range 0-3000000.")

        df[current_upb_col] = df[current_upb_col].astype("int32")
        print(f"    Final datatype of {current_upb_col}: {df[current_upb_col].dtype}")

        print(f"{current_upb_col} cleaning complete. Remaining NaNs: {df[current_upb_col].isna().sum()}")

        return df

    else:
        print(f"{current_upb_col} column not found. Skipping cleaning.")
        return df

# Function for cleaning 'CURRENT_DELINQUENCY_STATUS' column
def clean_delinquency_status(df):
    del_status_col = 'CURRENT_DELINQUENCY_STATUS'

    if del_status_col in df.columns:
        print(f"Cleaning {del_status_col} column...")

        df[del_status_col] = df[del_status_col].astype(str).str.strip().str.upper()

        df[f"{del_status_col}_IS_RA"] = (df[del_status_col] == 'RA').astype("int8")
        print(f"    Created '{del_status_col}_IS_RA' indicator column.")
        print(f"    '{del_status_col}_IS_RA' value counts:\n{df[f"{del_status_col}_IS_RA"].value_counts()}")

        nans_before_numeric_coercion = df[del_status_col].isna().sum()
        df[del_status_col] = pd.to_numeric(df[del_status_col], errors='coerce')

        df[f"{del_status_col}_IS_MISSING"] = df[del_status_col].isna().astype("int8")

        newly_coerced_nans_count = df[del_status_col].isna().sum() - nans_before_numeric_coercion
        if newly_coerced_nans_count > 0:
            print(f"    Converted 'RA' and {newly_coerced_nans_count} other non-numeric values to NaN.")
        else:
            print(f"    No new non-numeric values converted to NaN.")

        df.loc[df[f"{del_status_col}_IS_RA"] == 1, f"{del_status_col}_IS_MISSING"] = 0
        print(f"    Refined '{del_status_col}_IS_MISSING' to exclude 'RA' values (they are handled by '{del_status_col}_RA').")
        print(f"    '{del_status_col}_IS_MISSING' value counts:\n{df[f'{del_status_col}_IS_MISSING'].value_counts()}")

        median_del = df[del_status_col].median()
        print(f"    Calculated median {del_status_col} value: {median_del}")

        final_nan_count_before_fill = df[del_status_col].isna().sum()
        if final_nan_count_before_fill > 0:
            print(f"    Found {final_nan_count_before_fill} NaN values. Imputing with median ({median_del}).")
            df[del_status_col] = df[del_status_col].fillna(median_del)
        else:
            print(f"    No NaNs to impute in {del_status_col}.")

        max_del_cap = 18
        original_min = df[del_status_col].min()
        original_max = df[del_status_col].max()

        if original_min < 0 or original_max > max_del_cap:
            print(f"    Warning: Found {del_status_col} values outside of 0-{max_del_cap} ({original_min}-{original_max}). Clipping.")
            df[del_status_col] = np.clip(df[del_status_col], 0, max_del_cap)
        else:
            print(f"    All {del_status_col} in range 0-{max_del_cap}.")

        df[del_status_col] = df[del_status_col].astype("int8")
        print(f"    Final datatype of {del_status_col}: {df[del_status_col].dtype}")
        print(f"{del_status_col} cleaning complete. Remaining NaNs: {df[del_status_col].isna().sum()}")

        return df

    else:
        print(f"{del_status_col} column not found. Skipping cleaning.")
        return df

# Function for cleaning 'LOAN_AGE' column
def clean_loan_age(df):
    age_col = 'LOAN_AGE'

    if age_col in df.columns:
        print(f"Cleaning {age_col} column...")

        df[age_col] = pd.to_numeric(df[age_col], errors = 'coerce')

        negative_value_count = (df[age_col] < 0).sum()
        if negative_value_count > 0:
            print(f"    Found {negative_value_count} negative values. Converting to NaN.")
            df.loc[df[age_col] < 0, age_col] = np.nan
        else:
            print(f"    No negative values found in {age_col}.")

        df[f"{age_col}_IS_MISSING"] = df[age_col].isna().astype("int8")
        print(f"    Created '{age_col}_IS_MISSING' indicator column.")
        print(f"    '{age_col}_IS_MISSING' value counts:\n{df[f"{age_col}_IS_MISSING"].value_counts()}")

        median_age = df[age_col].median()
        print(f"    Calculated median {age_col} value: {median_age}")

        final_nan_count = df[age_col].isna().sum()
        if final_nan_count > 0:
            print(f"    Found {final_nan_count} NaN values. Imputing with median ({median_age}).")
            df[age_col] = df[age_col].fillna(median_age)
        else:
            print(f"    No NaN values to impute in {age_col}.")

        original_min = df[age_col].min()
        original_max = df[age_col].max()
        max_age = 600

        if original_min < 0 or original_max > max_age:
            print(f"    Warning: Found {age_col} values outside of 0-{max_age} ({original_min}-{original_max}). Clipping.")
            df[age_col] = np.clip(df[age_col], 0, max_age)
        else:
            print(f"    All {age_col} in range 0-{max_age}.")

        df[age_col] = df[age_col].astype("int16")
        print(f"    Final datatype of {age_col}: {df[age_col].dtype}")
        print(f"{age_col} cleaning complete. Remaining NaNs: {df[age_col].isna().sum()}")

        return df

    else:
        print(f"{age_col} column not found. Skipping cleaning.")
        return df

# Function for cleaning 'ZERO_BALANCE_EFFECT_DATE' column
def clean_zero_balance_date(df):
    zbe_col = 'ZERO_BALANCE_EFFECT_DATE'

    if zbe_col in df.columns:
        print(f"Cleaning {zbe_col} column...")

        df[zbe_col] = pd.to_datetime(df[zbe_col].astype(str), format="%Y%m", errors='coerce')
        print(f"    Converted {zbe_col} (YYYYMM format) to datetime.")

        df[f"{zbe_col}_IS_MISSING"] = df[zbe_col].isna().astype("int8")
        print(f"    Created '{zbe_col}_IS_MISSING' indicator column.")
        print(f"    '{zbe_col}_IS_MISSING' value counts:\n{df[f"{zbe_col}_IS_MISSING"].value_counts()}")

        initial_nat_count = df[zbe_col].isna().sum()
        if initial_nat_count > 0:
            mode_date = df[zbe_col].mode()[0]
            print(f"    Found {initial_nat_count} NaT values in {zbe_col}. Imputing with mode date: {mode_date.strftime('%Y-%m-%d')}")
            df[zbe_col] = df[zbe_col].fillna(mode_date)
        else:
            print(f"    No NaT values found in {zbe_col}.")

        print(f"    Final datatype of {zbe_col}: {df[zbe_col].dtype}")
        print(f"{zbe_col} cleaning complete. Remaining NaNs: {df[zbe_col].isna().sum()}")

        return df

    else:
        print(f"{zbe_col} column not found. Skipping cleaning.")
        return df

# Function for cleaning 'MODIFICATION_FLAG' column
def clean_modification_flag(df):
    mod_col = 'MODIFICATION_FLAG'

    if mod_col in df.columns:
        print(f"Cleaning {mod_col} column...")

        df[mod_col] = df[mod_col].astype(str).str.strip().str.upper()

        df[mod_col] = df[mod_col].map({'Y': 'Y', 'P': 'P'})
        print(f"    Mapped valid categories in {mod_col}.")

        df[f"{mod_col}_IS_MISSING"] = df[mod_col].isna().astype("int8")
        print(f"    Created '{mod_col}_IS_MISSING' indicator column.")
        print(f"    '{mod_col}_IS_MISSING' value counts:\n{df[f"{mod_col}_IS_MISSING"].value_counts()}")

        final_nan_count = df[mod_col].isna().sum()
        if final_nan_count > 0:
            print(f"    Found {final_nan_count} NaN values. Filling with 'NOT_MODIFIED'.")
            df[mod_col] = df[mod_col].fillna('NOT_MODIFIED')
        else:
            print(f"    No NaN values found in {mod_col}.")

        df[mod_col] = df[mod_col].astype('category')
        print(f"    {mod_col} unique values and counts after cleaning:\n{df[mod_col].value_counts(dropna=False)}")
        print(f"    Final datatype of {mod_col}: {df[mod_col].dtype}")

        print(f"{mod_col} cleaning complete. Remaining NaNs: {df[mod_col].isna().sum()}")

        return df

    else:
        print(f"{mod_col} column not found. Skipping cleaning.")
        return df

# Function for cleaning 'ZERO_BALANCE_CODE' column
def clean_zero_balance_code(df):
    zbc_col = 'ZERO_BALANCE_CODE'

    if zbc_col in df.columns:
        print(f"Cleaning {zbc_col} column...")

        df[zbc_col] = pd.to_numeric(df[zbc_col], errors='coerce')

        df[zbc_col] = df[zbc_col].map({
            1: 'PREPAID_OR_MATURED',
            2: 'THIRD_PARTY_SALE',
            3: 'SHORT_SALE_OR_CHARGE_OFF', 
            9: 'REO_DISPOSITION',
            15: 'WHOLE_LOAN_SALES',
            16: 'REPERFORMING_LOAN_SECURITIZATIONS',
            96: 'DEFECT_PRIOR_TO_TERMINATION' 
        })
        print(f"    Mapped valid categories in {zbc_col}.")

        df[f"{zbc_col}_IS_MISSING"] = df[zbc_col].isna().astype("int8")
        print(f"    Created '{zbc_col}_IS_MISSING' indicator column.")
        print(f"    '{zbc_col}_IS_MISSING' value counts:\n{df[f"{zbc_col}_IS_MISSING"].value_counts()}")

        final_nan_count = df[zbc_col].isna().sum()
        if final_nan_count > 0:
            print(f"    Found {final_nan_count} NaN values. Filling with 'NO_ZERO_BALANCE_CODE'.")
            df[zbc_col] = df[zbc_col].fillna('NO_ZERO_BALANCE_CODE')
        else:
            print(f"    No NaN values found in {zbc_col}.")

        df[zbc_col] = df[zbc_col].astype('category')
        print(f"    {zbc_col} unique values and counts after cleaning:\n{df[zbc_col].value_counts(dropna=False)}")
        print(f"    Final datatype of {zbc_col}: {df[zbc_col].dtype}")

        print(f"{zbc_col} cleaning complete. Remaining NaNs: {df[zbc_col].isna().sum()}")

        return df

    else:
        print(f"{zbc_col} column not found. Skipping cleaning.")
        return df

# Function for cleaning 'CURRENT_INTEREST_RATE' column
def clean_current_interest_rate(df):
    rate_col = 'CURRENT_INTEREST_RATE'

    if rate_col in df.columns:
        print(f"Cleaning {rate_col} column...")

        df[rate_col] = pd.to_numeric(df[rate_col], errors='coerce')

        initial_missing_count = (df[rate_col] == 0).sum()
        if initial_missing_count > 0:
            print(f"    Found {initial_missing_count} '{0}' values. Converting to NaN.")
            df[rate_col] = df[rate_col].replace(0, np.nan)
        else:
            print(f"    No '{0}' values found.")

        df[f"{rate_col}_IS_MISSING"] = df[rate_col].isna().astype("int8")
        print(f"    Created '{rate_col}_IS_MISSING' indicator column.")
        print(f"    '{rate_col}_IS_MISSING' value counts:\n{df[f"{rate_col}_IS_MISSING"].value_counts()}")

        median_rate = df[rate_col].median()
        print(f"    Calculated median {rate_col} value: {median_rate}")

        if df[rate_col].isna().any():
            df[rate_col] = df[rate_col].fillna(median_rate)
            print(f"    Imputed NaNs in {rate_col} with median ({median_rate})")
        else:
            print(f"    No NaNs to impute in {rate_col}.")

        original_min = df[rate_col].min()
        original_max = df[rate_col].max()
        if original_min < 0.5 or original_max > 20.0:
            print(f"    Warning: {rate_col} values detected outside 0.5-20.0 range ({original_min}-{original_max}). Clipping.")
            df[rate_col] = np.clip(df[rate_col], 0.5, 20.0)
        else:
            print(f"    {rate_col} values are within 0.5-20.0 range.")

        df[rate_col] = df[rate_col].astype("float32")
        print(f"    Final datatype of {rate_col}: {df[rate_col].dtype}")

        print(f"{rate_col} cleaning complete. Remaining NaNs: {df[rate_col].isna().sum()}")

        return df
    else:
        print(f"{rate_col} column not found. Skipping cleaning.")
        return df

# Function for cleaning 'CURRENT_NON_INTEREST_UPB' column
def clean_current_non_interest_upb(df):
    nonint_upb_col = 'CURRENT_NON_INTEREST_UPB'

    if nonint_upb_col in df.columns:
        print(f"Cleaning {nonint_upb_col} column...")

        df[nonint_upb_col] = pd.to_numeric(df[nonint_upb_col], errors='coerce')

        df[f"{nonint_upb_col}_IS_MISSING"] = df[nonint_upb_col].isna().astype("int8")
        print(f"    Created '{nonint_upb_col}_IS_MISSING' indicator column.")
        print(f"    '{nonint_upb_col}_IS_MISSING' value counts:\n{df[f"{nonint_upb_col}_IS_MISSING"].value_counts()}")

        median_upb = df[nonint_upb_col].median()
        print(f"    Calculated median {nonint_upb_col} value: {median_upb}")

        final_missing_count = df[nonint_upb_col].isna().sum()
        if final_missing_count > 0:
            print(f"    Found {final_missing_count} NaN values. Imputing with median ({median_upb}).")
            df[nonint_upb_col] = df[nonint_upb_col].fillna(median_upb)
        else:
            print(f"    No NaN values found in {nonint_upb_col}.")

        original_min = df[nonint_upb_col].min()
        original_max = df[nonint_upb_col].max()

        if original_min < 0 or original_max > 3000000:
            print(f"    Warning: Found {nonint_upb_col} values outside of 0-3000000 ({original_min}-{original_max}). Clipping.")
            df[nonint_upb_col] = np.clip(df[nonint_upb_col], 0, 3000000)
        else:
            print(f"    All {nonint_upb_col} in range 0-3000000.")

        df[nonint_upb_col] = df[nonint_upb_col].astype("int32")
        print(f"    Final datatype of {nonint_upb_col}: {df[nonint_upb_col].dtype}")

        print(f"{nonint_upb_col} cleaning complete. Remaining NaNs: {df[nonint_upb_col].isna().sum()}")

        return df

    else:
        print(f"{nonint_upb_col} column not found. Skipping cleaning.")
        return df

# Function for cleaning 'STEP_MOD_FLAG' column
def clean_step_mod_flag(df):
    step_mod_col = 'STEP_MOD_FLAG'

    if step_mod_col in df.columns:
        print(f"Cleaning {step_mod_col} column...")

        df[step_mod_col] = df[step_mod_col].astype(str).str.strip().str.upper()

        df[step_mod_col] = df[step_mod_col].map({'Y': 'Y', 'N': 'N'})
        print(f"    Mapped {step_mod_col} to valid categories.")

        df[f"{step_mod_col}_IS_MISSING"] = df[step_mod_col].isna().astype("int8")
        print(f"    Created '{step_mod_col}_IS_MISSING' indicator column.")
        print(f"    '{step_mod_col}_IS_MISSING' value counts:\n{df[f"{step_mod_col}_IS_MISSING"].value_counts()}")

        final_nan_count = df[step_mod_col].isna().sum()
        if final_nan_count > 0:
            print(f"    Found {final_nan_count} NaN values. Filling with 'NOT_MODIFIED_IN_PERIOD'.")
            df[step_mod_col] = df[step_mod_col].fillna('NOT_MODIFIED_IN_PERIOD')
        else:
            print(f"    No NaN values found in {step_mod_col}.")

        df[step_mod_col] = df[step_mod_col].astype('category')
        print(f"    {step_mod_col} unique values and counts after cleaning:\n{df[step_mod_col].value_counts(dropna=False)}")
        print(f"    Final datatype of {step_mod_col}: {df[step_mod_col].dtype}")

        print(f"{step_mod_col} cleaning complete. Remaining NaNs: {df[step_mod_col].isna().sum()}")

        return df

    else:
        print(f"{step_mod_col} column not found. Skipping cleaning.")
        return df

# Function for cleaning 'PAYMENT_DEFERRAL' column
def clean_payment_deferral(df):
    pay_defer_col = 'PAYMENT_DEFERRAL'

    if pay_defer_col in df.columns:
        print(f"Cleaning {pay_defer_col} column...")

        df[pay_defer_col] = df[pay_defer_col].astype(str).str.strip().str.upper()

        df[pay_defer_col] = df[pay_defer_col].map({'Y': 'Y', 'P': 'P'})
        print(f"    Mapped {pay_defer_col} to valid categories.")

        df[f"{pay_defer_col}_IS_MISSING"] = df[pay_defer_col].isna().astype("int8")
        print(f"    Created '{pay_defer_col}_IS_MISSING' indicator column.")
        print(f"    '{pay_defer_col}_IS_MISSING' value counts:\n{df[f"{pay_defer_col}_IS_MISSING"].value_counts()}")

        final_nan_count = df[pay_defer_col].isna().sum()
        if final_nan_count > 0:
            print(f"    Found {final_nan_count} NaN values. Filling with 'NOT_PAYMENT_DEFERRAL'.")
            df[pay_defer_col] = df[pay_defer_col].fillna('NOT_PAYMENT_DEFERRAL')
        else:
            print(f"    No NaN values found in {pay_defer_col}.")

        df[pay_defer_col] = df[pay_defer_col].astype('category')
        print(f"    {pay_defer_col} unique values and counts after cleaning:\n{df[pay_defer_col].value_counts(dropna=False)}")
        print(f"    Final datatype of {pay_defer_col}: {df[pay_defer_col].dtype}")

        print(f"{pay_defer_col} cleaning complete. Remaining NaNs: {df[pay_defer_col].isna().sum()}")

        return df

    else:
        print(f"{pay_defer_col} column not found. Skipping cleaning.")
        return df

# Function for cleaning 'ELTV' column