from typing import Any
import numpy as np

# --- TSON Format Constants ---
TSON_FORMAT_IDENTIFIER = "tson"
MINIMUM_ITEMS_FOR_COMPRESSION = 3  # Balance: overhead vs compression benefit

class TypeInference:
    """Maps Python types to TSON schema type identifiers."""
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STRING = "str"
    
    @staticmethod
    def inferFromValue(value: Any) -> str:
        """
        Infer TSON type string from a Python value.
        
        Returns: Type identifier string for schema generation.
        """
        if isinstance(value, bool):  # Must check bool before int (bool is subclass of int)
            return TypeInference.BOOL
        if isinstance(value, (int, np.integer)):
            return TypeInference.INT
        if isinstance(value, (float, np.floating)):
            return TypeInference.FLOAT
        return TypeInference.STRING

def isUniformDictList(data_list: list) -> bool:
    """
    Check if the list contains enough dictionaries to justify TSON.
    
    Now more lenient: as long as they are all dicts, we can force 
    uniformity by using the superset of all keys.
    """
    if not data_list or len(data_list) < MINIMUM_ITEMS_FOR_COMPRESSION:
        return False
    return all(isinstance(item, dict) for item in data_list)

def convertToTSON(data_list: list) -> dict | list:
    """
    Convert a list of dicts to TSON, even if keys are non-uniform (sparse).
    
    It calculates the superset of all keys to ensure a complete schema,
    filling missing values with None.
    """
    if not isUniformDictList(data_list):
        return data_list
    
    # Calculate union of all keys to handle sparse dicts (where nulls were dropped)
    all_keys_set = set()
    for item in data_list:
        all_keys_set.update(item.keys())
    
    # Sort keys for deterministic output and matching schema
    fieldOrder = sorted(list(all_keys_set))
    
    # Build schema - using the first available value for each key for type inference
    schema = []
    for k in fieldOrder:
        # Find first non-None value for type inference
        sample_val = next((item[k] for item in data_list if k in item and item[k] is not None), None)
        typeString = TypeInference.inferFromValue(sample_val)
        schema.append(f"{k}:{typeString}")
    
    # Build data rows with None for missing keys
    dataRows = [
        [item.get(k) for k in fieldOrder]
        for item in data_list
    ]
    
    return {
        "format": TSON_FORMAT_IDENTIFIER,
        "schema": schema,
        "data": dataRows
    }