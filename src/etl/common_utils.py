from pyspark.sql.functions import (
    lit, when, udf
)
from pyspark.sql.types import StringType

# UDF for parsing MSA codes
@udf(returnType=StringType())
def _parse_msa_code_udf(value: str) -> str:
    if value is None:
        return None
    value_stripped_upper = value.strip().upper()

    if (value_stripped_upper == 'NON_MSA_OR_UNKNOWN'):
        return value_stripped_upper

    if value_stripped_upper == 'NAN' or value_stripped_upper == '':
        return None
    try:
        return str(int(float()))
    except:
        return None

# UDF for postal code validation and formatting
@udf(returnType=StringType())
def _postal_code_validator_udf(value: str) -> str:
    if value is None:
        return None
    value_stripped = value.strip()

    if value_stripped.upper() == 'UNKNOWN_POSTAL_CODE':
        return value_stripped.upper()

    if value_stripped.lower() == 'nan' or value_stripped == '':
        return None
    
    try:
        value_num = int(float(value_stripped))

        if value_num % 100 != 0:
            return None
        
        if value_num < 100 or value_num > 99900:
            return None
        
        return '{:05d}'.format(value_num)
    except:
        return None

# UDF for Number of Borrowers string conversion
@udf(returnType=StringType())
def _clean_numeric_string_for_map_udf(s: str) -> str:
    if s is None:
        return None
    s_stripped = s.strip()
    if s_stripped.lower() == 'nan' or s_stripped == '':
        return None
    try:
        num_val =float(s_stripped)
        if num_val == 99.0:
            return None
        return str(int(num_val))
    except:
        return None

# Helper function for building conditional expressions
def build_categorical_map_exp(column_expr, mapping_dict, default_fill_values=None):
    column_expr_str = column_expr.cast(StringType())

    case_stmt = None

    # 1. Map cleaned values to themselves
    for cleaned_val in set(mapping_dict.values()): # Use set to get unique output values
        if case_stmt is None:
            case_stmt = when(column_expr_str == lit(cleaned_val), lit(cleaned_val).cast(StringType()))
        else:
            case_stmt = case_stmt.when(column_expr_str == lit(cleaned_val), lit(cleaned_val).cast(StringType()))
    
    # 2. Map keys to cleaned values
    for raw_val, cleaned_val in mapping_dict.items():
        if case_stmt is None: # This path should not be taken if mapping_dict has entries.
            case_stmt = when(column_expr_str == lit(raw_val), lit(cleaned_val).cast(StringType()))
        else:
            case_stmt = case_stmt.when(column_expr_str == lit(raw_val), lit(cleaned_val).cast(StringType()))

    # 3. Default for values not in keys or cleaned vals
    if default_fill_values is not None:
        return case_stmt.otherwise(lit(default_fill_values).cast(StringType()))
    else:
        return case_stmt.otherwise(lit(None).cast(StringType()))