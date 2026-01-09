import pandas as pd
import numpy as np

# Function for dopping unnecessary columns in origination dataset
def drop_origination_columns(df):
    cols_to_drop = ['PRE_RELIEF_LOAN_NUMBER']
    return df.drop(columns=cols_to_drop)

# Function for deriving the origination date from LOAN_SEQUENCE_NUMBER, creating the 'ORIGINATION_DATE' column.
def derive_orig_date_from_loan_num(df):
    loan_seq_col = 'LOAN_SEQUENCE_NUMBER'
    orig_date_col = 'ORIGINATION_DATE'

    if loan_seq_col not in df.columns:
        print(f"Error: {loan_seq_col} column not found. Cannot derive {orig_date_col}.")
        return df

    print(f"Deriving {orig_date_col} from {loan_seq_col}...")

    # Helper function for parsing single loan sequence number string
    def _parse_loan_seq_to_date_string(loan_seq_str):
        if not isinstance(loan_seq_str, str) or len(loan_seq_str) < 4:
            return np.nan

        try:
            year_suffix = loan_seq_str[1:3]
            quarter_num = int(loan_seq_str[4])

            if year_suffix == '99':
                full_year = 1999
            else:
                full_year = 2000 + int(year_suffix)

            month_map = {1: 1, 2: 4, 3: 7, 4: 10}
            month = month_map[quarter_num]

            date_str = f"{full_year}-{month:02d}-01"
            return date_str
        except (ValueError, KeyError, IndexError):
            return np.nan

    df[orig_date_col] = df[loan_seq_col].astype(str).apply(_parse_loan_seq_to_date_string)
    print(f"    Created '{orig_date_col}' column.")

    df[orig_date_col] = pd.to_datetime(df[orig_date_col], errors='coerce')
    print(f"    Final datatype of {orig_date_col}: {df[orig_date_col].dtype}")

    initial_nat_count = df[orig_date_col].isna().sum()
    if initial_nat_count > 0:
        print(f"    Found {initial_nat_count} NaT values in {orig_date_col} after derivation.")
        mode_date = df[orig_date_col].mode()[0]
        print(f"    Imputing NaTs in {orig_date_col} with mode date: {mode_date.strftime('%Y-%m-%d')}.")
        df[orig_date_col] = df[orig_date_col].fillna(mode_date)
    else:
        print(f"    No NaT values found in {orig_date_col}.")

    print(f"{orig_date_col} derivation complete. Final datatype: {df[orig_date_col].dtype}.")
    return df

# Function for cleaning 'CREDIT_SCORE' column
def clean_credit_score(df):
    credit_score_col = 'CREDIT_SCORE'
    if credit_score_col in df.columns:
        print(f"Cleaning {credit_score_col} column...")

        initial_missing_count = (df[credit_score_col] == 9999).sum()
        if initial_missing_count > 0:
            print(f"    Found {initial_missing_count} '{9999}' values. Converting to NaN.")
            df[credit_score_col] = df[credit_score_col].replace(9999, np.nan)
        else:
            print(f"    No '{9999}' values found, checking for existing NaNs.")
    
        df[f"{credit_score_col}_IS_MISSING"] = df[credit_score_col].isna().astype("int8")
        print(f"    Created '{credit_score_col}_IS_MISSING' indicator column")
        print(f"    '{credit_score_col}_IS_MISSING' value counts:\n{df[f'{credit_score_col}_IS_MISSING'].value_counts()}")

        median_score = df[credit_score_col].median()
        print(f"    Calculated median {credit_score_col}: {median_score}")

        if df[credit_score_col].isna().any():
            df[credit_score_col] = df[credit_score_col].fillna(median_score)
            print(f"    Imputed missing values in {credit_score_col} with median ({median_score}).")
        else:
            print(f"    No NaNs to impute in {credit_score_col}")

        original_min = df[credit_score_col].min()
        original_max = df[credit_score_col].max()
        if original_min < 300 or original_max > 850:
            print(f"    Warning: {credit_score_col} values detected outside of 300-850 range ({original_min}-{original_max}). Clipping.")
            df[credit_score_col] = np.clip(df[credit_score_col], 300, 850)
        else:
            print(f"    {credit_score_col} values are within 300-850 range.")

        df[credit_score_col] = df[credit_score_col].astype("int16")
        print(f"    Final datatype of {credit_score_col}: {df[credit_score_col].dtype}")

        print(f"{credit_score_col} cleaning complete. Remaining NaNs: {df[credit_score_col].isna().sum()}")

        return df
    
    else:
        print(f"{credit_score_col} column not found. Skipping cleaning.")
        return df

# Function for cleaning 'FIRST_TIME_HOMEBUYER_FLAG' column
def clean_first_homebuyer_flag(df):
    homebuyer_flag_col = 'FIRST_TIME_HOMEBUYER_FLAG'

    if homebuyer_flag_col in df.columns:
        print(f"Cleaning {homebuyer_flag_col} column...")

        df[homebuyer_flag_col] = df[homebuyer_flag_col].astype(str)

        initial_missing_count = (df[homebuyer_flag_col] == '9').sum()
        if initial_missing_count > 0:
            print(f"    Found {initial_missing_count} '{9}' values. Converting to NaN.")
            df[homebuyer_flag_col] = df[homebuyer_flag_col].replace('9', np.nan)
        else:
            print(f"    No '{9}' values found, checking for existing NaNs.")

        df[homebuyer_flag_col] = df[homebuyer_flag_col].str.strip()

        print(f"    {homebuyer_flag_col} values stripped of whitespace.")

        df[homebuyer_flag_col] = df[homebuyer_flag_col].map({'Y': 1, 'N': 0}).fillna(0).astype('int8')

        print(f"    Final datatype of {homebuyer_flag_col}: {df[homebuyer_flag_col].dtype}")
        print(f"{homebuyer_flag_col} cleaning complete. Remaining NaNs: {df[homebuyer_flag_col].isna().sum()}")

        return df
    else:
        print(f"{homebuyer_flag_col} column not found. Skipping cleaning.")
        return df

# Function for coverting Metropolitan Division / MSA Codes to numeric strings (helper function)
def convert_to_numeric_msa(value):
    if isinstance(value, str) and value.lower() == 'nan':
        return np.nan
    if pd.isna(value):
        return np.nan
    if value == 'NON_MSA_OR_UNKNOWN':
        return value
    try:
        return str(int(float(value)))
    except (ValueError, TypeError):
        return 'OTHER_INVALID_CODE'

# Function for cleaning 'MSA_OR_MET_DIV' column
def clean_met_msa_code(df):
    met_msa_col = 'MSA_OR_MET_DIV'

    if met_msa_col in df.columns:
        print(f"Cleaning {met_msa_col} column...")
        
        initial_missing_count = df[met_msa_col].isna().sum()
        if initial_missing_count > 0:
            print(f"    Found {initial_missing_count} initial NaN values.")
        else:
            print(f"    No initial NaN values found in {met_msa_col}.")
        
        df[met_msa_col] = df[met_msa_col].astype(str)

        df[met_msa_col] = df[met_msa_col].str.strip()
        print(f"    {met_msa_col} values stripped of whitespace.")

        print("    Applying custom conversion for numeric formats and invalid codes...")
        df[met_msa_col] = df[met_msa_col].apply(convert_to_numeric_msa)
        print(f"    Ensured {met_msa_col} are numeric strings or defined categories.")

        final_nan_count_after_conversion = df[met_msa_col].isna().sum()
        if final_nan_count_after_conversion > 0:
            print(f"    Found {final_nan_count_after_conversion} NaNs (after custom conversion). Filling with 'NON_MSA_OR_UNKNOWN'.")
            df[met_msa_col] = df[met_msa_col].fillna('NON_MSA_OR_UNKNOWN')
        else:
            print(f"    No NaNs to fill with 'NON_MSA_OR_UNKNOWN'.")

        df[met_msa_col] = df[met_msa_col].astype('category')
        print(f"    {met_msa_col} unique values and counts after cleaning:\n{df[met_msa_col].value_counts(dropna=False)}")
        print(f"    Final datatype of {met_msa_col}: {df[met_msa_col].dtype}")

        print(f"{met_msa_col} cleaning complete. Remaining NaNs: {df[met_msa_col].isna().sum()}")

        if 'OTHER_INVALID_CODE' in df[met_msa_col].cat.categories:
             print(f"Note: 'OTHER_INVALID_CODE' count: {df[met_msa_col].value_counts().get('OTHER_INVALID_CODE', 0)}")

        return df
    else:
        print(f"{met_msa_col} column not found. Skipping cleaning.")
        return df

# Function for cleaning 'MI_PERCENT' column
def clean_mi_percent(df):
    mi_percent_col = 'MI_PERCENT'

    if mi_percent_col in df.columns:
        print(f"Cleaing {mi_percent_col} column...")

        initial_nan_count = (df[mi_percent_col] == 999).sum()
        if initial_nan_count > 0:
            print(f"    Found {initial_nan_count} '{999}' values. Converting to NaN.")
            df[mi_percent_col] = df[mi_percent_col].replace(999, np.nan)
        else:
            print(f"    No '{999}' values found.")

        df[f"{mi_percent_col}_IS_MISSING"] = df[mi_percent_col].isna().astype("int8")
        print(f"    Created '{mi_percent_col}_IS_MISSING' indicator column")
        print(f"    '{mi_percent_col}_IS_MISSING' value counts:\n{df[f'{mi_percent_col}_IS_MISSING'].value_counts()}")

        median_mi = df[mi_percent_col].median()
        print(f"    Calculated median {mi_percent_col}: {median_mi}")

        if df[mi_percent_col].isna().any():
            df[mi_percent_col] = df[mi_percent_col].fillna(median_mi)
            print(f"    Imputed missing values in {mi_percent_col} with median ({median_mi}).")
        else:
            print(f"    No NaNs to impute in {mi_percent_col}")

        current_min = df[mi_percent_col].min()
        current_max = df[mi_percent_col].max()
        if current_min < 0 or current_max > 55:
            print(f"    Warning: {mi_percent_col} values detected outside 0-55 range ({current_min}-{current_max}). Clipping.")
            df[mi_percent_col] = np.clip(df[mi_percent_col], 0, 55)
        else:
            print(f"    {mi_percent_col} values are within 0-55 range.")

        df[mi_percent_col] = df[mi_percent_col].astype("int8")
        print(f"    Final datatype of {mi_percent_col}: {df[mi_percent_col].dtype}")

        print(f"{mi_percent_col} cleaning complete. Remaining NaNs: {df[mi_percent_col].isna().sum()}")

        return df
    else:
        print(f"{mi_percent_col} column not found. Skipping cleaning.")
        return df

# Function for cleaning 'NUMBER_OF_UNITS' column
def clean_num_units(df):
    num_units_col = 'NUMBER_OF_UNITS'

    if num_units_col in df.columns:
        print(f"Cleaning {num_units_col} column...")

        df[num_units_col] = pd.to_numeric(df[num_units_col], errors='coerce')

        initial_missing_count = (df[num_units_col] == 99).sum()
        if initial_missing_count > 0:
            print(f"    Found {initial_missing_count} '{99}' values. Converting to NaN.")
            df[num_units_col] = df[num_units_col].replace(99, np.nan)
        else:
            print(f"    No '{99}' values found.")

        df[f"{num_units_col}_IS_MISSING"] = df[num_units_col].isna().astype("int8")
        print(f"    Created '{num_units_col}_IS_MISSING' indicator column.")
        print(f"    '{num_units_col}_IS_MISSING' value counts:\n{df[f"{num_units_col}_IS_MISSING"].value_counts()}")

        if df[num_units_col].isna().any():
            df[num_units_col] = df[num_units_col].fillna("UNITS_UNKNOWN")
            print(f"    Filled missing values in {num_units_col} with 'UNITS_UNKNOWN'.")
        else:
            print(f"    No NaNs to fill in {num_units_col}.")

        df[num_units_col] = df[num_units_col].astype('category')
        print(f"    {num_units_col} unique values and counts after cleaning:\n{df[num_units_col].value_counts(dropna=False)}")
        print(f"    Final datatype of {num_units_col}: {df[num_units_col].dtype}")

        print(f"{num_units_col} cleaning complete. Remaining NaNs: {df[num_units_col].isna().sum()}")

        return df
    else:
        print(f"{num_units_col} column not found. Skipping cleaning.")
        return df

# Function for cleaning 'OCCUPANCY_STATUS' column
def clean_occ_status(df):
    occ_status_col = 'OCCUPANCY_STATUS'

    if occ_status_col in df.columns:
        print(f"Cleaning {occ_status_col} column...")

        df[occ_status_col] = df[occ_status_col].astype(str)
        print(f"    Converted {occ_status_col} datatype to string.")

        initial_missing_values = (df[occ_status_col] == '9').sum()
        if initial_missing_values > 0:
            print(f"    Found {initial_missing_values} '{9}' values. Converting to NaN.")
            df[occ_status_col] = df[occ_status_col].replace('9', np.nan)
        else:
            print(f"    No '9' values found.")

        df[f"{occ_status_col}_IS_MISSING"] = df[occ_status_col].isna().astype("int8")
        print(f"    Created '{occ_status_col}_IS_MISSING' indicator column.")
        print(f"    '{occ_status_col}_IS_MISSING' value counts:\n{df[f"{occ_status_col}_IS_MISSING"].value_counts()}")

        if df[occ_status_col].isna().any():
            df[occ_status_col] = df[occ_status_col].fillna('OCCUPANCY_UNKNOWN')
            print(f"    Filled missing values in {occ_status_col} with 'OCCUPANCY_UNKNOWN'.")
        else:
            print(f"    No NaNs to fill in {occ_status_col}.")

        df[occ_status_col] = df[occ_status_col].str.strip().str.upper()
        
        df[occ_status_col] = df[occ_status_col].astype('category')
        print(f"    {occ_status_col} unique values and counts after cleaning:\n{df[occ_status_col].value_counts(dropna=False)}")
        print(f"    Final datatype of {occ_status_col}: {df[occ_status_col].dtype}")

        print(f"{occ_status_col} cleaning complete. Remaining NaNs: {df[occ_status_col].isna().sum()}")

        return df
    else:
        print(f"{occ_status_col} column not found. Skipping cleaning.")
        return df

# Function for cleaning 'ORIGINAL_CLTV' column (Must be done after creating the 'ORIGINATION_DATE' column)
def clean_original_cltv(df):
    cltv_col = 'ORIGINAL_CLTV'

    if "ORIGINATION_DATE" not in df.columns or not pd.api.types.is_datetime64_any_dtype(df["ORIGINATION_DATE"]):
        print(f"Warning: 'ORIGINATION_DATE' column not found or not in datetime format. Cannot perform date-dependent cleaning for {cltv_col}.")
        print("Please ensure 'ORIGINATION_DATE' is cleaned and converted to datetime before calling this function.")
        return df

    if cltv_col in df.columns:
        print(f"Cleaning {cltv_col} column...")

        df[cltv_col] = pd.to_numeric(df[cltv_col], errors='coerce')

        initial_missing_values = (df[cltv_col] == 999).sum()
        if initial_missing_values > 0:
            print(f"    Found {initial_missing_values} '{999}' values. Converting to NaN.")
            df[cltv_col] = df[cltv_col].replace(999, np.nan)
        else:
            print(f"    No '{999}' values found.")
        
        df[f"{cltv_col}_IS_MISSING"] = df[cltv_col].isna().astype("int8")
        print(f"    Created '{cltv_col}_IS_MISSING' indicator column.")
        print(f"    '{cltv_col}_IS_MISSING' value counts:\n{df[f"{cltv_col}_IS_MISSING"].value_counts()}")

        median_cltv = df[cltv_col].median()
        print(f"    Calculated median {cltv_col}: {median_cltv}")

        if df[cltv_col].isna().any():
            df[cltv_col] = df[cltv_col].fillna(median_cltv)
            print(f"    Imputed NaNs in {cltv_col} with median ({median_cltv}).")
        else:
            print(f"    No NaNs to impute in {cltv_col}.")

        split_date = pd.to_datetime('2018-03-31')

        mask_2018q1_prior = df["ORIGINATION_DATE"] <= split_date
        mask_2018_later = df["ORIGINATION_DATE"] > split_date

        print(f"    Applying date-dependent clipping for {cltv_col}...")

        if not df.loc[mask_2018q1_prior].empty:
            original_min_prior = df.loc[mask_2018q1_prior, cltv_col].min()
            original_max_prior = df.loc[mask_2018q1_prior, cltv_col].max()
            print(f"    Loans originated <= {split_date.strftime('%Y-%m-%d')}: Range {original_min_prior} - {original_max_prior}. Clipping to [6, 200].")
            df.loc[mask_2018q1_prior, cltv_col] = np.clip(df.loc[mask_2018q1_prior, cltv_col], 6, 200)
        else:
            print(f"    No loans originated <= {split_date.strftime('%Y-%m-%d')} found.")

        if not df.loc[mask_2018_later].empty:
            original_min_later = df.loc[mask_2018_later, cltv_col].min()
            original_max_later = df.loc[mask_2018_later, cltv_col].max()
            print(f"    Loans originated > {split_date.strftime('%Y-%m-%d')}: Range {original_min_later} - {original_max_later}. Clipping to [1, 998].")
            df.loc[mask_2018_later, cltv_col] = np.clip(df.loc[mask_2018_later, cltv_col], 1, 998)
        else:
            print(f"    No loans originated > {split_date.strftime('%Y-%m-%d')} found.")

        df[cltv_col] = df[cltv_col].astype("int16")
        print(f"    Final datatype of {cltv_col}: {df[cltv_col].dtype}")
        
        print(f"{cltv_col} cleaning complete. Remaining NaNs: {df[cltv_col].isna().sum()}")

        return df
    else:
        print(f"{cltv_col} column not found. Skipping cleaning.")
        return df

# Function for cleaning 'ORIGINAL_DTI_RATIO' column
def clean_dti(df):
    dti_col = 'ORIGINAL_DTI_RATIO'

    if dti_col in df.columns:
        print(f"Cleaning {dti_col} column...")

        df[dti_col] = pd.to_numeric(df[dti_col], errors='coerce')

        initial_missing_count = (df[dti_col] == 999).sum()
        if initial_missing_count > 0:
            print(f"    Found {initial_missing_count} '{999}' values. Converting to NaN.")
            df[dti_col] = df[dti_col].replace(999, np.nan)
        else:
            print(f"    No '{999}' values found.")

        df[f"{dti_col}_IS_MISSING"] = df[dti_col].isna().astype("int8")
        print(f"    Created '{dti_col}_IS_MISSING' indicator column.")
        print(f"    '{dti_col}_IS_MISSING' value counts:\n{df[f"{dti_col}_IS_MISSING"].value_counts()}")

        median_dti = df[dti_col].median()
        print(f"    Calculated median {dti_col} value: {median_dti}")

        if df[dti_col].isna().any():
            df[dti_col] = df[dti_col].fillna(median_dti)
            print(f"    Imputed NaNs in {dti_col} with median ({median_dti}).")
        else:
            print(f"    No NaNs to impute in {dti_col}.")

        original_min = df[dti_col].min()
        original_max = df[dti_col].max()

        if original_min < 1 or original_max > 65:
            print(f"    Warning: {dti_col} values detected outside of 1-65 range ({original_min}-{original_max}). Clipping.")
            df[dti_col] = np.clip(df[dti_col], 1, 65)
        else:
            print(f"    {dti_col} values are within 1-65 range.")

        df[dti_col] = df[dti_col].astype("int8")
        print(f"    Final datatype of {dti_col}: {df[dti_col].dtype}")

        print(f"{dti_col} cleaning complete. Remaining NaNs: {df[dti_col].isna().sum()}")

        return df
    else:
        print(f"{dti_col} column not found. Skipping cleaning.")
        return df

# Function for cleaning 'ORIGINAL_UPB' column
def clean_original_upb(df):
    upb_col = 'ORIGINAL_UPB'

    if upb_col in df.columns:
        print(f"Cleaning {upb_col} column...")

        df[upb_col] = pd.to_numeric(df[upb_col], errors='coerce')

        initial_missing_count = (df[upb_col] == 0).sum()
        if initial_missing_count > 0:
            print(f"    Found {initial_missing_count} '{0}' values. Converting to NaN.")
            df[upb_col] = df[upb_col].replace(0, np.nan)
        else:
            print(f"    No '{0}' values found.")
        
        df[f"{upb_col}_IS_MISSING"] = df[upb_col].isna().astype("int8")
        print(f"    Created '{upb_col}_IS_MISSING' indicator column.")
        print(f"    '{upb_col}_IS_MISSING' value counts:\n{df[f"{upb_col}_IS_MISSING"].value_counts()}")

        median_upb = df[upb_col].median()
        print(f"    Calculated median {upb_col} value: {median_upb}")

        if df[upb_col].isna().any():
            df[upb_col] = df[upb_col].fillna(median_upb)
            print(f"    Imputed NaNs in {upb_col} with median ({median_upb}).")
        else:
            print(f"    No NaNs to impute in {upb_col}.")

        original_min = df[upb_col].min()
        original_max = df[upb_col].max()
        if original_min < 1 or original_max > 3000000:
            print(f"    Warning: {upb_col} values detected outside of 1-3000000 range ({original_min}-{original_max}). Clipping.")
            df[upb_col] = np.clip(df[upb_col], 1, 3000000)
        else:
            print(f"    {upb_col} values are within 1-3000000 range.")

        df[upb_col] = df[upb_col].astype("int32")
        print(f"    Final datatype of {upb_col}: {df[upb_col].dtype}")

        print(f"{upb_col} cleaning complete. Remaining NaNs: {df[upb_col].isna().sum()}")

        return df
    else:
        print(f"{upb_col} column not found. Skipping cleaning.")
        return df

# Function for cleaning 'ORIGINAL_LTV' column (Must be done after creating the 'ORIGINATION_DATE' column)
def clean_original_ltv(df):
    ltv_col = 'ORIGINAL_LTV'

    if "ORIGINATION_DATE" not in df.columns or not pd.api.types.is_datetime64_any_dtype(df["ORIGINATION_DATE"]):
        print(f"Warning: 'ORIGINATION_DATE' column not found or not in datetime format. Cannot perform date-dependent cleaning for {ltv_col}.")
        print("Please ensure 'ORIGINATION_DATE' is cleaned and converted to datetime before calling this function.")
        return df

    if ltv_col in df.columns:
        print(f"Cleaning {ltv_col} column...")

        df[ltv_col] = pd.to_numeric(df[ltv_col], errors='coerce')

        initial_missing_values = (df[ltv_col] == 999).sum()
        if initial_missing_values > 0:
            print(f"    Found {initial_missing_values} '{999}' values. Converting to NaN.")
            df[ltv_col] = df[ltv_col].replace(999, np.nan)
        else:
            print(f"    No '{999}' values found.")
        
        df[f"{ltv_col}_IS_MISSING"] = df[ltv_col].isna().astype("int8")
        print(f"    Created '{ltv_col}_IS_MISSING' indicator column.")
        print(f"    '{ltv_col}_IS_MISSING' value counts:\n{df[f"{ltv_col}_IS_MISSING"].value_counts()}")

        median_ltv = df[ltv_col].median()
        print(f"    Calculated median {ltv_col}: {median_ltv}")

        if df[ltv_col].isna().any():
            df[ltv_col] = df[ltv_col].fillna(median_ltv)
            print(f"    Imputed NaNs in {ltv_col} with median ({median_ltv}).")
        else:
            print(f"    No NaNs to impute in {ltv_col}.")

        split_date = pd.to_datetime('2018-03-31')

        mask_2018q1_prior = df["ORIGINATION_DATE"] <= split_date
        mask_2018_later = df["ORIGINATION_DATE"] > split_date

        print(f"    Applying date-dependent clipping for {ltv_col}...")

        if not df.loc[mask_2018q1_prior].empty:
            original_min_prior = df.loc[mask_2018q1_prior, ltv_col].min()
            original_max_prior = df.loc[mask_2018q1_prior, ltv_col].max()
            print(f"    Loans originated <= {split_date.strftime('%Y-%m-%d')}: Range {original_min_prior} - {original_max_prior}. Clipping to [6, 105].")
            df.loc[mask_2018q1_prior, ltv_col] = np.clip(df.loc[mask_2018q1_prior, ltv_col], 6, 105)
        else:
            print(f"    No loans originated <= {split_date.strftime('%Y-%m-%d')} found.")

        if not df.loc[mask_2018_later].empty:
            original_min_later = df.loc[mask_2018_later, ltv_col].min()
            original_max_later = df.loc[mask_2018_later, ltv_col].max()
            print(f"    Loans originated > {split_date.strftime('%Y-%m-%d')}: Range {original_min_later} - {original_max_later}. Clipping to [1, 998].")
            df.loc[mask_2018_later, ltv_col] = np.clip(df.loc[mask_2018_later, ltv_col], 1, 998)
        else:
            print(f"    No loans originated > {split_date.strftime('%Y-%m-%d')} found.")

        df[ltv_col] = df[ltv_col].astype("int16")
        print(f"    Final datatype of {ltv_col}: {df[ltv_col].dtype}")
        
        print(f"{ltv_col} cleaning complete. Remaining NaNs: {df[ltv_col].isna().sum()}")

        return df
    else:
        print(f"{ltv_col} column not found. Skipping cleaning.")
        return df

# Function for cleaning 'ORIGINAL_INTEREST_RATE' column
def clean_original_interest_rate(df):
    rate_col = 'ORIGINAL_INTEREST_RATE'

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

# Function for cleaning 'CHANNEL' column
def clean_channel(df):
    channel_col = 'CHANNEL'

    if channel_col in df.columns:
        print(f"Cleaning {channel_col} column...")

        df[channel_col] = df[channel_col].astype(str)

        initial_missing_count = (df[channel_col] == '9').sum()
        if initial_missing_count > 0:
            print(f"    Found {initial_missing_count} '9' values. Converting to NaNs.")
            df[channel_col] = df[channel_col].replace('9', np.nan)
        else:
            print(f"    No '9' values found.")

        df[f"{channel_col}_IS_MISSING"] = df[channel_col].isna().astype("int8")
        print(f"    Created '{channel_col}_IS_MISSING' indicator column.")
        print(f"    '{channel_col}_IS_MISSING' value counts:\n{df[f"{channel_col}_IS_MISSING"].value_counts()}")

        if df[channel_col].isna().any():
            df[channel_col] = df[channel_col].fillna('CHANNEL_UNKNOWN')
            print(f"    Filled NaN values with 'CHANNEL_UNKNOWN'")
        else:
            print(f"    No NaN values found in {channel_col}")

        df[channel_col] = df[channel_col].str.strip().str.upper()
        print(f"    Stripped values of whitespace and made uppercase.")

        df[channel_col] = df[channel_col].astype('category')
        print(f"    {channel_col} unique values and counts after cleaning:\n{df[channel_col].value_counts(dropna=False)}")
        print(f"    Final datatype of {channel_col}: {df[channel_col].dtype}")

        print(f"{channel_col} cleaning complete. Remaining NaNs: {df[channel_col].isna().sum()}")

        return df

    else:
        print(f"{channel_col} column not found. Skipping cleaning.")
        return df

# Function for cleaning 'PPM_FLAG' column
def clean_ppm_flag(df):
    ppm_col = 'PPM_FLAG'

    if ppm_col in df.columns:
        print(f"Cleaning {ppm_col} column...")

        df[ppm_col] = df[ppm_col].astype(str)

        df[ppm_col] = df[ppm_col].str.strip().str.upper()
        print(f"    Stripped values of whitespace and made uppercase.")

        df[ppm_col] = df[ppm_col].map({'Y': 1, 'N': 0})
        print(f"    Mapped 'Y' values to 1 and 'N' values to 0")

        df[f"{ppm_col}_IS_MISSING"] = df[ppm_col].isna().astype("int8")
        print(f"    Created '{ppm_col}_IS_MISSING' indicator column.")
        print(f"    '{ppm_col}_IS_MISSING' value counts:\n{df[f"{ppm_col}_IS_MISSING"].value_counts()}")

        missing_count = df[ppm_col].isna().sum()
        if missing_count > 0:
            print(f"    Found {missing_count} missing values. Filling with 0.")
            df[ppm_col] = df[ppm_col].fillna(0)
        else:
            print(f"    No missing values found.")

        df[ppm_col] = df[ppm_col].astype("int8")
        print(f"    Final datatype of {ppm_col}: {df[ppm_col].dtype}")
        print(f"{ppm_col} cleaning complete. Remaining NaNs: {df[ppm_col].isna().sum()}")

        return df

    else:
        print(f"{ppm_col} column not found. Skipping cleaning.")
        return df

# Funcation for cleaning 'AMORTIZATION_TYPE' column
def clean_amortization_type(df):
    amortization_col = "AMORTIZATION_TYPE"

    if amortization_col in df.columns:
        print(f"Cleaning {amortization_col} column...")

        df[amortization_col] = df[amortization_col].astype(str)
        df[amortization_col] = df[amortization_col].str.strip().str.upper()
        print(f"    Stripped values of whitespace and made uppercase.")

        expected_valid_types = ['FRM', 'ARM']
        df[f"{amortization_col}_IS_MISSING"] = (~df[amortization_col].isin(expected_valid_types)).astype("int8")
        print(f"    Created '{amortization_col}_IS_MISSING' indicator column.")
        print(f"    '{amortization_col}_IS_MISSING' value counts:\n{df[f"{amortization_col}_IS_MISSING"].value_counts()}")

        mapping = {'FRM': 'FRM', 'ARM': 'ARM'}
        df[amortization_col] = df[amortization_col].map(mapping)

        if df[amortization_col].isna().any():
            print(f"    Found NaNs (from unmapped values). Filling with 'AMORTIZATION_TYPE_UNKNOWN'.")
            df[amortization_col] = df[amortization_col].fillna('AMORTIZATION_TYPE_UNKNOWN')
        else:
            print(f"    No NaNs to fill with 'AMORTIZATION_TYPE_UNKNOWN'.")

        df[amortization_col] = df[amortization_col].astype('category')
        print(f"    {amortization_col} unique values and counts after cleaning:\n{df[amortization_col].value_counts(dropna=False)}")
        print(f"    Final datatype of {amortization_col}: {df[amortization_col].dtype}")

        print(f"{amortization_col} cleaning complete. Remaining NaNs: {df[amortization_col].isna().sum()}")

        return df

    else:
        print(f"{amortization_col} column not found. Skipping cleaning.")
        return df

# Function for cleaning 'PROPERTY_STATE' column
def clean_property_state(df):
    state_col = 'PROPERTY_STATE'

    if state_col in df.columns:
        print(f"Cleaning {state_col} column...")

        US_STATES_AND_TERRITORIES_CANONICAL = {
            'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA',
            'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM',
            'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'PR', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA',
            'VI', 'WA', 'WV', 'WI', 'WY', 'AS', 'FM', 'GU', 'MH', 'MP', 'PW'
        }

        df[state_col] = df[state_col].astype(str).str.strip().str.upper()
        print(f"    Stripped values of whitespace and made uppercase.")

        df[f"{state_col}_IS_MISSING"] = (~df[state_col].isin(US_STATES_AND_TERRITORIES_CANONICAL)).astype("int8")
        print(f"    Created '{state_col}_IS_MISSING' indicator column.")
        print(f"    '{state_col}_IS_MISSING' value counts:\n{df[f"{state_col}_IS_MISSING"].value_counts()}")

        mapping = {s: s for s in US_STATES_AND_TERRITORIES_CANONICAL}
        df[state_col] = df[state_col].map(mapping)

        if df[state_col].isna().any():
            print(f"    Found NaNs (from unmapped values). Filling with 'STATE_OR_TERRITORY_UNKNOWN'.")
            df[state_col] = df[state_col].fillna('STATE_OR_TERRITORY_UNKNOWN')
        else:
            print(f"    No NaNs to fill with 'STATE_OR_TERRITORY_UNKNOWN'.")

        df[state_col] = df[state_col].astype('category')
        print(f"    {state_col} unique values and counts after cleaning:\n{df[state_col].value_counts(dropna=False)}")
        print(f"    Final datatype of {state_col}: {df[state_col].dtype}")

        print(f"{state_col} cleaning complete. Remaining NaNs: {df[state_col].isna().sum()}")

        return df

    else:
        print(f"{state_col} column not found. Skipping cleaning.")
        return df

# Function for cleaning 'PROPERTY_TYPE' column
def clean_property_type(df):
    prop_type_col = 'PROPERTY_TYPE'

    if prop_type_col in df.columns:
        print(f"Cleaning {prop_type_col} column...")

        df[prop_type_col] = df[prop_type_col].astype(str).str.strip().str.upper()

        mapping = {'CO': 'CO', 'PU': 'PU', 'MH': 'MH', 'SF': 'SF', 'CP': 'CP'}
        df[prop_type_col] = df[prop_type_col].map(mapping)
        print(f"    Mapped expected values and set others to NaN.")

        df[f"{prop_type_col}_IS_MISSING"] = df[prop_type_col].isna().astype("int8")
        print(f"    Created '{prop_type_col}_IS_MISSING' indicator column.")
        print(f"    '{prop_type_col}_IS_MISSING' value counts:\n{df[f"{prop_type_col}_IS_MISSING"].value_counts()}")

        if df[prop_type_col].isna().any():
            print(f"    Found NaNs (from unmapped values). Filling with 'PROPERTY_TYPE_UNKNOWN'")
            df[prop_type_col] = df[prop_type_col].fillna('PROPERTY_TYPE_UNKNOWN')
        else:
            print(f"    No NaNs to fill with 'PROPERTY_TYPE_UNKNOWN'.")

        df[prop_type_col] = df[prop_type_col].astype("category")
        print(f"    {prop_type_col} unique values and counts after cleaning:\n{df[prop_type_col].value_counts(dropna=False)}")
        print(f"    Final datatype of {prop_type_col}: {df[prop_type_col].dtype}")

        print(f"{prop_type_col} cleaning complete. Remaining NaNs: {df[prop_type_col].isna().sum()}")

        return df

    else:
        print(f"{prop_type_col} column not found. Skipping cleaning.")
        return df

# Helper function for POSTAL_CODE column
def validate_and_format_postal_code(code_val):
    if isinstance(code_val, str) and (code_val.lower() == 'nan' or code_val.strip() == ''):
        return np.nan

    if pd.isna(code_val):
        return np.nan

    try:
        num_code = int(float(code_val))

        if num_code % 100 != 0:
            return np.nan

        if num_code < 100 or num_code > 99900:
            return np.nan

        return '{:05d}'.format(num_code)
    except (ValueError, TypeError):
        return np.nan

# Function for cleaning 'POSTAL_CODE' column
def clean_postal_code(df):
    postal_col = 'POSTAL_CODE'

    if postal_col in df.columns:
        print(f"Cleaning {postal_col} column...")

        df[postal_col] = pd.to_numeric(df[postal_col], errors = 'coerce')

        initial_zero_count = (df[postal_col] == 0).sum()
        if initial_zero_count > 0:
            print(f"    Found {initial_zero_count} '0' values. Converting to NaN.")
            df[postal_col] = df[postal_col].replace(0, np.nan)
        else:
            print(f"    No '0' values found.")

        df[postal_col] = df[postal_col].astype(str).str.strip()
        print(f"    Stripped {postal_col} of whitespace and converted to string.")

        print(f"    Applying format validation and zero-padding for {postal_col}...")
        nans_before_validation = df[postal_col].isna().sum()
        df[postal_col] = df[postal_col].apply(validate_and_format_postal_code)
        nans_after_validation = df[postal_col].isna().sum()
        invalid_formats_count = nans_after_validation - nans_before_validation
        if invalid_formats_count > 0:
            print(f"    Found {invalid_formats_count} with invalid format or out of range. Converting to NaN.")
        else:
            print(f"    No new invalid formats or out-of-range values found by validation.")

        df[f"{postal_col}_IS_MISSING"] = df[postal_col].isna().astype("int8")
        print(f"    Created '{postal_col}_IS_MISSING' indicator column.")
        print(f"    '{postal_col}_IS_MISSING' value counts:\n{df[f"{postal_col}_IS_MISSING"].value_counts()}")

        final_nan_count = df[postal_col].isna().sum()
        if final_nan_count > 0:
            print(f"    Found {final_nan_count} NaNs (total after all invalidations). Filling with 'POSTAL_CODE_UNKNOWN'.")
            df[postal_col] = df[postal_col].fillna('POSTAL_CODE_UNKNOWN')
        else:
            print(f"    No NaNs to fill with 'POSTAL_CODE_UNKNOWN'.")

        df[postal_col] = df[postal_col].astype('category')
        print(f"    {postal_col} unique values and counts after cleaning:\n{df[postal_col].value_counts(dropna=False)}")
        print(f"    Final datatype of {postal_col}: {df[postal_col].dtype}")

        print(f"{postal_col} cleaning complete. Remaining NaNs: {df[postal_col].isna().sum()}")

        return df

    else:
        print(f"{postal_col} column not found. Skipping cleaning.")
        return df

# Function for validating 'LOAN_SEQUENCE_NUMBER' column
def validate_loan_sequence_number_originations(df):
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

        if not df[loan_number].is_unique:
            duplicate_count = df[loan_number].duplicated().sum()
            print(f"    Warning: {duplicate_count} duplicate values found in {loan_number}. These indicate duplicate loan entries.")
            df.drop_duplicates(subset=[loan_number], inplace=True)
            print(f"    Dropped duplicate {loan_number} entries. Remaining rows: {len(df)}")
        else:
            print(f"    No duplicate values found in {loan_number}.")

        print(f"{loan_number} validation complete.")

        return df
    else:
        print(f"{loan_number} column not found. Skipping validation.")
        return df

# Function for cleaning 'LOAN_PURPOSE' column
def clean_loan_purpose(df):
    purpose_col = 'LOAN_PURPOSE'

    if purpose_col in df.columns:
        print(f"Cleaning {purpose_col} column...")

        df[purpose_col] = df[purpose_col].astype(str).str.strip().str.upper()

        mapping = {'P': 'P', 'N': 'N', 'C': 'C', 'R': 'R'}
        df[purpose_col] = df[purpose_col].map(mapping)
        print(f"    Mapped expected values and set others to NaN.")

        df[f"{purpose_col}_IS_MISSING"] = df[purpose_col].isna().astype("int8")
        print(f"    Created '{purpose_col}_IS_MISSING' indicator column.")
        print(f"    '{purpose_col}_IS_MISSING' value counts:\n{df[f"{purpose_col}_IS_MISSING"].value_counts()}")

        if df[purpose_col].isna().any():
            print(f"    Found NaNs (from unmapped values). Filling with 'LOAN_PURPOSE_UNKNOWN'")
            df[purpose_col] = df[purpose_col].fillna('LOAN_PURPOSE_UNKNOWN')
        else:
            print(f"    No NaNs to fill with 'LOAN_PURPOSE_UNKNOWN'.")

        df[purpose_col] = df[purpose_col].astype("category")
        print(f"    {purpose_col} unique values and counts after cleaning:\n{df[purpose_col].value_counts(dropna=False)}")
        print(f"    Final datatype of {purpose_col}: {df[purpose_col].dtype}")
        print(f"{purpose_col} cleaning complete. Remaining NaNs: {df[purpose_col].isna().sum()}")

        return df

    else:
        print(f"{purpose_col} column not found. Skipping cleaning.")
        return df

# Function for cleaning 'ORIGINAL_LOAN_TERM' column
def clean_loan_term(df):
    loan_term_col = 'ORIGINAL_LOAN_TERM'

    if loan_term_col in df.columns:
        print(f"Cleaning {loan_term_col} column...")

        df[loan_term_col] = pd.to_numeric(df[loan_term_col], errors='coerce')

        initial_zero_count = (df[loan_term_col] == 0).sum()
        if initial_zero_count > 0:
            print(f"    Found {initial_zero_count} '0' values. Converting to NaN.")
            df[loan_term_col] = df[loan_term_col].replace(0, np.nan)
        else:
            print(f"    No '0' values found.")

        df[f"{loan_term_col}_IS_MISSING"] = df[loan_term_col].isna().astype("int8")
        print(f"    Created '{loan_term_col}_IS_MISSING' indicator column.")
        print(f"    '{loan_term_col}_IS_MISSING' value counts:\n{df[f"{loan_term_col}_IS_MISSING"].value_counts()}")

        median_term = df[loan_term_col].median()
        print(f"    Calculated median {loan_term_col} value: {median_term}")

        if df[loan_term_col].isna().any():
            print(f"    Imputing NaN values with median ({median_term}).")
            df[loan_term_col] = df[loan_term_col].fillna(median_term)
        else:
            print(f"    No NaN values to impute.")

        original_min = df[loan_term_col].min()
        original_max = df[loan_term_col].max()
        if original_min < 60 or original_max > 480:
            print(f"    Warning: {loan_term_col} values found outside of 60-480 range ({original_min}-{original_max}). Clipping.")
            df[loan_term_col] = np.clip(df[loan_term_col], 60, 480)
        else:
            print(f"    {loan_term_col} values are within 60-480 range.")

        df[loan_term_col] = df[loan_term_col].astype("int16")
        print(f"    Final datatype of {loan_term_col}: {df[loan_term_col].dtype}")

        print(f"{loan_term_col} cleaning complete. Remaining NaNs: {df[loan_term_col].isna().sum()}")

        return df

    else:
        print(f"{loan_term_col} column not found. Skipping cleaning.")
        return df

# Function for cleaning 'NUMBER_OF_BORROWERS' column (Must be done after creating the 'ORIGINATION_DATE' column)
def clean_num_borrowers(df):
    num_borr_col = 'NUMBER_OF_BORROWERS'
    orig_date_col = 'ORIGINATION_DATE'

    if orig_date_col not in df.columns or not pd.api.types.is_datetime64_any_dtype(df[orig_date_col]):
        print(f"Error: {orig_date_col} column not found or not in datetime format. Cannot perform date-dependent cleaning for {num_borr_col}.")
        print(f"Please ensure '{orig_date_col}' is cleaned and converted to datetime before calling this function.")
        return df

    if num_borr_col in df.columns:
        print(f"Cleaning {num_borr_col} column...")

        df[num_borr_col] = pd.to_numeric(df[num_borr_col], errors='coerce')

        split_date = pd.to_datetime('2018-03-31')
        mask_2018q1_prior = df[orig_date_col] <= split_date
        mask_2018_later = df[orig_date_col] > split_date

        processed_borr_series = pd.Series(np.nan, index=df.index, dtype=object)

        print(f"    Applying date-dependent mapping for {num_borr_col}...")

        if not df.loc[mask_2018q1_prior].empty:
            mapping_prior = {'1': '1_BORROWER', '2': '>1_BORROWERS_OLD_RULE'}
            processed_borr_series.loc[mask_2018q1_prior] = df.loc[mask_2018q1_prior, num_borr_col].astype(str).map(mapping_prior)
            print(f"    Processed {mask_2018q1_prior.sum()} loans <= {split_date.strftime('%Y-%m-%d')}.")
        else:
            print(f"    No loans originated <= {split_date.strftime('%Y-%m-%d')} found.")

        if not df.loc[mask_2018_later].empty:
            mapping_later = {str(i): str(i) for i in range(1,11)}
            processed_borr_series.loc[mask_2018_later] = df.loc[mask_2018_later, num_borr_col].astype(str).map(mapping_later)
            print(f"    Processed {mask_2018_later.sum()} loans > {split_date.strftime('%Y-%m-%d')}.")
        else:
            print(f"    No loans originated > {split_date.strftime('%Y-%m-%d')} found.")

        df[num_borr_col] = processed_borr_series

        df[f"{num_borr_col}_IS_MISSING"] = df[num_borr_col].isna().astype("int8")
        print(f"    Created '{num_borr_col}_IS_MISSING' indicator column.")
        print(f"    '{num_borr_col}_IS_MISSING' value counts:\n{df[f"{num_borr_col}_IS_MISSING"].value_counts()}")

        final_nan_count = df[num_borr_col].isna().sum()
        if final_nan_count > 0:
            print(f"    Found {final_nan_count} NaNs (total after all processing). Filling with 'NUM_BORROWERS_UNKNOWN'.")
            df[num_borr_col] = df[num_borr_col].fillna('NUM_BORROWERS_UNKNOWN')
        else:
            print(f"    No NaNs to fill with 'NUM_BORROWERS_UNKNOWN'.")

        df[num_borr_col] = df[num_borr_col].astype('category')
        print(f"    {num_borr_col} unique values and counts after cleaning:\n{df[num_borr_col].value_counts(dropna=False)}")
        print(f"    Final datatype of {num_borr_col}: {df[num_borr_col].dtype}")

        print(f"{num_borr_col} cleaning complete. Remaining NaNs: {df[num_borr_col].isna().sum()}")

        return df

    else:
        print(f"{num_borr_col} column not found. Skipping cleaning.")
        return df

# Function for cleaning 'SUPER_CONFORMING_FLAG' column
def clean_super_conforming_flag(df):
    super_conform_col = 'SUPER_CONFORMING_FLAG'

    if super_conform_col in df.columns:
        print(f"Cleaning {super_conform_col} column...")

        df[super_conform_col] = df[super_conform_col].astype(str).str.strip().str.upper()

        df[super_conform_col] = df[super_conform_col].map({'Y': 1})
        print(f"    Mapped Y values to 1, all others are NaN.")

        final_nan_count = df[super_conform_col].isna().sum()
        if final_nan_count > 0:
            print(f"    Found {final_nan_count} NaNs in {super_conform_col}. Converting to '0'.")
            df[super_conform_col] = df[super_conform_col].fillna(0)
        else:
            print(f"    Found no NaNs to convert.")

        df[super_conform_col] = df[super_conform_col].astype("int8")
        print(f"    Final datatype of {super_conform_col}: {df[super_conform_col].dtype}")

        print(f"{super_conform_col} cleaning complete. Remaining NaNs: {df[super_conform_col].isna().sum()}")

        return df

    else:
        print(f"{super_conform_col} column not found. Skipping cleaning.")
        return df

# Function for cleaning 'PROGRAM_INDICATOR' column
def clean_program_indicator(df):
    program_col = 'PROGRAM_INDICATOR'

    if program_col in df.columns:
        print(f"Cleaning {program_col} column...")

        df[program_col] = df[program_col].astype(str).str.strip().str.upper()

        mapping = {'H': 'H', 'F': 'F', 'R': 'R'}
        df[program_col] = df[program_col].map(mapping)
        print(f"    Mapped expected values and set others to NaN.")

        df[f"{program_col}_IS_MISSING"] = df[program_col].isna().astype("int8")
        print(f"    Created '{program_col}_IS_MISSING' indicator column.")
        print(f"    '{program_col}_IS_MISSING' value counts:\n{df[f"{program_col}_IS_MISSING"].value_counts()}")

        if df[program_col].isna().any():
            print(f"    Found NaNs (from unmapped values). Filling with 'PROGRAM_INDICATOR_UNKNOWN'")
            df[program_col] = df[program_col].fillna('PROGRAM_INDICATOR_UNKNOWN')
        else:
            print(f"    No NaNs to fill with 'PROGRAM_INDICATOR_UNKNOWN'.")

        df[program_col] = df[program_col].astype("category")
        print(f"    {program_col} unique values and counts after cleaning:\n{df[program_col].value_counts(dropna=False)}")
        print(f"    Final datatype of {program_col}: {df[program_col].dtype}")

        print(f"{program_col} cleaning complete. Remaining NaNs: {df[program_col].isna().sum()}")

        return df

    else:
        print(f"{program_col} column not found. Skipping cleaning.")
        return df

# Function for cleaning 'PROP_VALUATION_METHOD' column
def clean_prop_val_method(df):
    prop_val_col = 'PROP_VALUATION_METHOD'

    if prop_val_col in df.columns:
        print(f"Cleaning {prop_val_col} column...")

        df[prop_val_col] = df[prop_val_col].astype(str).str.strip().str.upper()

        mapping = {'1': '1', '2': '2', '3': '3', '4': '4'}
        df[prop_val_col] = df[prop_val_col].map(mapping)
        print(f"    Mapped expected values and set others to NaN.")

        df[f"{prop_val_col}_IS_MISSING"] = df[prop_val_col].isna().astype("int8")
        print(f"    Created '{prop_val_col}_IS_MISSING' indicator column.")
        print(f"    '{prop_val_col}_IS_MISSING' value counts:\n{df[f"{prop_val_col}_IS_MISSING"].value_counts()}")

        if df[prop_val_col].isna().any():
            print(f"    Found NaNs (from unmapped values). Filling with 'PROP_VAL_METHOD_UNKNOWN'")
            df[prop_val_col] = df[prop_val_col].fillna('PROP_VAL_METHOD_UNKNOWN')
        else:
            print(f"    No NaNs to fill with 'PROP_VAL_METHOD_UNKNOWN'.")

        df[prop_val_col] = df[prop_val_col].astype("category")
        print(f"    {prop_val_col} unique values and counts after cleaning:\n{df[prop_val_col].value_counts(dropna=False)}")
        print(f"    Final datatype of {prop_val_col}: {df[prop_val_col].dtype}")

        print(f"{prop_val_col} cleaning complete. Remaining NaNs: {df[prop_val_col].isna().sum()}")

        return df

    else:
        print(f"{prop_val_col} column not found. Skipping cleaning.")
        return df

# Function for cleaning 'IO_INDICATOR' column
def clean_io_indicator(df):
    io_flag_col = 'IO_INDICATOR'

    if io_flag_col in df.columns:
        print(f"Cleaning {io_flag_col} column...")

        df[io_flag_col] = df[io_flag_col].astype(str).str.strip().str.upper()

        print(f"    {io_flag_col} values stripped of whitespace.")

        df[io_flag_col] = df[io_flag_col].map({'Y': 1, 'N': 0}).fillna(0).astype('int8')

        print(f"    Final datatype of {io_flag_col}: {df[io_flag_col].dtype}")
        print(f"{io_flag_col} cleaning complete. Remaining NaNs: {df[io_flag_col].isna().sum()}")

        return df
    else:
        print(f"{io_flag_col} column not found. Skipping cleaning.")
        return df

# Function for cleaning 'MI_CANCEL_INDICATOR' column
def clean_mi_indicator(df):
    mi_flag_col = 'MI_CANCEL_INDICATOR'

    if mi_flag_col in df.columns:
        print(f"Cleaning {mi_flag_col} column...")

        df[mi_flag_col] = df[mi_flag_col].astype(str).str.strip().str.upper()

        mapping = {'Y': 'Y', 'N': 'N', '7': '7', '9': '9'}
        df[mi_flag_col] = df[mi_flag_col].map(mapping)
        print(f"    Mapped expected values and set others to NaN.")

        df[f"{mi_flag_col}_IS_MISSING"] = df[mi_flag_col].isna().astype("int8")
        print(f"    Created '{mi_flag_col}_IS_MISSING' indicator column.")
        print(f"    '{mi_flag_col}_IS_MISSING' value counts:\n{df[f"{mi_flag_col}_IS_MISSING"].value_counts()}")

        df[f"{mi_flag_col}_NOT_APPLICABLE"] = (df[mi_flag_col] == '7').astype("int8")
        print(f"    Created '{mi_flag_col}_NOT_APPLICABLE' indicator column.")
        print(f"    '{mi_flag_col}_NOT_APPLICABLE' value counts:\n{df[f"{mi_flag_col}_NOT_APPLICABLE"].value_counts()}")

        df[f"{mi_flag_col}_NOT_DISCLOSED"] = (df[mi_flag_col] == '9').astype("int8")
        print(f"    Created '{mi_flag_col}_NOT_DISCLOSED' indicator column.")
        print(f"    '{mi_flag_col}_NOT_DISCLOSED' value counts:\n{df[f"{mi_flag_col}_NOT_DISCLOSED"].value_counts()}")

        if df[mi_flag_col].isna().any():
            print(f"    Found NaNs (from unmapped values). Filling with 'MI_CANCEL_INDICATOR_UNKNOWN'")
            df[mi_flag_col] = df[mi_flag_col].fillna('MI_CANCEL_INDICATOR_UNKNOWN')
        else:
            print(f"    No NaNs to fill with 'MI_CANCEL_INDICATOR_UNKNOWN'.")

        df[mi_flag_col] = df[mi_flag_col].astype("category")
        print(f"    {mi_flag_col} unique values and counts after cleaning:\n{df[mi_flag_col].value_counts(dropna=False)}")
        print(f"    Final datatype of {mi_flag_col}: {df[mi_flag_col].dtype}")

        print(f"{mi_flag_col} cleaning complete. Remaining NaNs: {df[mi_flag_col].isna().sum()}")

        return df

    else:
        print(f"{mi_flag_col} column not found. Skipping cleaning.")
        return df

# Function for cleaning 'FIRST_PAYMENT_DATE' column
def clean_first_payment_date(df):
    fpd_col = 'FIRST_PAYMENT_DATE'

    if fpd_col in df.columns:
        print(f"Cleaning {fpd_col} column...")

        df[fpd_col] = pd.to_datetime(df[fpd_col].astype(str), format="%Y%m", errors='coerce')
        print(f"    Converted {fpd_col} (YYYYMM format) to datetime.")

        df[f"{fpd_col}_IS_MISSING"] = df[fpd_col].isna().astype("int8")
        print(f"    Created '{fpd_col}_IS_MISSING' indicator column.")
        print(f"    '{fpd_col}_IS_MISSING' value counts:\n{df[f"{fpd_col}_IS_MISSING"].value_counts()}")

        initial_nat_count = df[fpd_col].isna().sum()
        if initial_nat_count > 0:
            mode_date = df[fpd_col].mode()[0]
            print(f"    Found {initial_nat_count} NaT values in {fpd_col}. Imputing with mode date: {mode_date.strftime('%Y-%m-%d')}")
            df[fpd_col] = df[fpd_col].fillna(mode_date)
        else:
            print(f"    No NaT values found in {fpd_col}.")

        print(f"    Final datatype of {fpd_col}: {df[fpd_col].dtype}")
        print(f"{fpd_col} cleaning complete. Remaining NaNs: {df[fpd_col].isna().sum()}")

        return df

    else:
        print(f"{fpd_col} column not found. Skipping cleaning.")
        return df

# Function for cleaning 'MATURITY_DATE' column
def clean_maturity_date(df):
    md_col = 'MATURITY_DATE'

    if md_col in df.columns:
        print(f"Cleaning {md_col} column...")

        df[md_col] = pd.to_datetime(df[md_col].astype(str), format="%Y%m", errors='coerce')
        print(f"    Converted {md_col} (YYYYMM format) to datetime.")

        df[f"{md_col}_IS_MISSING"] = df[md_col].isna().astype("int8")
        print(f"    Created '{md_col}_IS_MISSING' indicator column.")
        print(f"    '{md_col}_IS_MISSING' value counts:\n{df[f"{md_col}_IS_MISSING"].value_counts()}")

        initial_nat_count = df[md_col].isna().sum()
        if initial_nat_count > 0:
            mode_date = df[md_col].mode()[0]
            print(f"    Found {initial_nat_count} NaT values in {md_col}. Imputing with mode date: {mode_date.strftime('%Y-%m-%d')}")
            df[md_col] = df[md_col].fillna(mode_date)
        else:
            print(f"    No NaT values found in {md_col}.")

        print(f"    Final datatype of {md_col}: {df[md_col].dtype}")
        print(f"{md_col} cleaning complete. Remaining NaNs: {df[md_col].isna().sum()}")

        return df

    else:
        print(f"{md_col} column not found. Skipping cleaning.")
        return df

# Function for cleaning 'SELLER_NAME' column
def clean_seller_name(df):
    seller_col = 'SELLER_NAME'

    if seller_col in df.columns:
        print(f"Cleaning {seller_col} column...")

        df[seller_col] = df[seller_col].astype(str).str.strip().str.upper()
        print(f"    Stripped whitespace and converted {seller_col} values to uppercase.")

        missing_placeholders = {'NAN', ''}

        mask_is_missing = df[seller_col].isin(missing_placeholders)

        df[f"{seller_col}_IS_MISSING"] = mask_is_missing.astype("int8")
        print(f"    Created '{seller_col}_IS_MISSING' indicator column.")
        print(f"    '{seller_col}_IS_MISSING' value counts:\n{df[f'{seller_col}_IS_MISSING'].value_counts()}")

        if mask_is_missing.any():
            print(f"    Found {mask_is_missing.sum()} missing/invalid seller names. Filling with 'UNKNOWN_SELLER'.")
            df.loc[mask_is_missing, seller_col] = 'UNKNOWN_SELLER'
        else:
            print(f"    No missing/invalid seller names found.")

        df[seller_col] = df[seller_col].astype('category')
        print(f"    {seller_col} unique values and counts after cleaning:\n{df[seller_col].value_counts(dropna=False)}")
        print(f"    Final datatype of {seller_col}: {df[seller_col].dtype}")
        print(f"{seller_col} cleaning complete. Remaining NaNs: {df[seller_col].isna().sum()}")

        return df

    else:
        print(f"{seller_col} column not found. Skipping cleaning.")
        return df

# Function for cleaning 'SERVICER_NAME' column
def clean_servicer_name(df):
    servicer_col = 'SERVICER_NAME'

    if servicer_col in df.columns:
        print(f"Cleaning {servicer_col} column...")

        df[servicer_col] = df[servicer_col].astype(str).str.strip().str.upper()
        print(f"    Stripped whitespace and converted {servicer_col} values to uppercase.")

        missing_placeholders = {'NAN', ''}

        mask_is_missing = df[servicer_col].isin(missing_placeholders)

        df[f"{servicer_col}_IS_MISSING"] = mask_is_missing.astype("int8")
        print(f"    Created '{servicer_col}_IS_MISSING' indicator column.")
        print(f"    '{servicer_col}_IS_MISSING' value counts:\n{df[f'{servicer_col}_IS_MISSING'].value_counts()}")

        if mask_is_missing.any():
            print(f"    Found {mask_is_missing.sum()} missing/invalid servicer names. Filling with 'UNKNOWN_SERVICER'.")
            df.loc[mask_is_missing, servicer_col] = 'UNKNOWN_SERVICER'
        else:
            print(f"    No missing/invalid servicer names found.")

        df[servicer_col] = df[servicer_col].astype('category')
        print(f"    {servicer_col} unique values and counts after cleaning:\n{df[servicer_col].value_counts(dropna=False)}")
        print(f"    Final datatype of {servicer_col}: {df[servicer_col].dtype}")
        print(f"{servicer_col} cleaning complete. Remaining NaNs: {df[servicer_col].isna().sum()}")

        return df

    else:
        print(f"{servicer_col} column not found. Skipping cleaning.")
        return df

# Function for cleaning 'RELIEF_REFI_INDICATOR' column
def clean_relief_refi_indicator(df):
    refi_col = 'RELIEF_REFI_INDICATOR'

    if refi_col in df.columns:
        print(f"Cleaning {refi_col} column...")

        df[refi_col] = df[refi_col].astype(str).str.strip().str.upper()

        print(f"    {refi_col} values stripped of whitespace.")

        df[refi_col] = df[refi_col].map({'Y': 1}).fillna(0).astype('int8')

        print(f"    Final datatype of {refi_col}: {df[refi_col].dtype}")
        print(f"{refi_col} cleaning complete. Remaining NaNs: {df[refi_col].isna().sum()}")

        return df
    else:
        print(f"{refi_col} column not found. Skipping cleaning.")
        return df

# Master cleaning function for originations data
def clean_originations_df(raw_df):
    # Drop columns
    df = drop_origination_columns(raw_df)
    # Validate Loan Sequence Number
    df = validate_loan_sequence_number_originations(df)
    # Create ORIGINATION_DATE column
    df = derive_orig_date_from_loan_num(df)
    # All other cleaning functions
    print("Beginning cleaning of originations dataset.")
    df = clean_credit_score(df)
    df = clean_first_homebuyer_flag(df)
    df = clean_met_msa_code(df)
    df = clean_mi_percent(df)
    df = clean_num_units(df)
    df = clean_occ_status(df)
    df = clean_original_cltv(df)
    df = clean_dti(df)
    df = clean_original_upb(df)
    df = clean_original_ltv(df)
    df = clean_original_interest_rate(df)
    df = clean_channel(df)
    df = clean_ppm_flag(df)
    df = clean_amortization_type(df)
    df = clean_property_state(df)
    df = clean_property_type(df)
    df = clean_postal_code(df)
    df = clean_loan_purpose(df)
    df = clean_loan_term(df)
    df = clean_num_borrowers(df)
    df = clean_super_conforming_flag(df)
    df = clean_program_indicator(df)
    df = clean_prop_val_method(df)
    df = clean_io_indicator(df)
    df = clean_mi_indicator(df)
    df = clean_first_payment_date(df)
    df = clean_maturity_date(df)
    df = clean_seller_name(df)
    df = clean_servicer_name(df)
    df = clean_relief_refi_indicator(df)
    print("Cleaning complete!")
    return df
