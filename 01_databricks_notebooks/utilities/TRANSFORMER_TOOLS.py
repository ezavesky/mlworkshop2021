# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.ml import Pipeline, PipelineModel

def transformer_load(path_primary, path_secondary=None):
    """
    This function attempts to load a saved transformer, first from a primary then a secondary path.
    It's used in this workshop to load from personal paths or workshop paths, if not available.  
    """
    try:
        list_files = dbutils.fs.ls(path_primary)
    except Exception as e:
        if path_secondary is None:
            fn_log(f"Failed to load transformer from '{path_primary}', no secondary provided, aborting...")
            return None
        fn_log(f"Primary failed, attempting to load secondary transformer '{path_secondary}'...")
        return transformer_load(path_secondary)
    pipe_loaded = Pipeline.read().load(path_primary)
    return pipe_loaded
  
def pipeline_model_load(path_primary, path_secondary=None):
    """
    This function attempts to load a saved transformer, first from a primary then a secondary path.
    It's used in this workshop to load from personal paths or workshop paths, if not available.  
    """
    try:
        list_files = dbutils.fs.ls(path_primary)
    except Exception as e:
        if path_secondary is None:
            fn_log(f"Failed to load transformer from '{path_primary}', no secondary provided, aborting...")
            return None
        fn_log(f"Primary failed, attempting to load secondary transformer '{path_secondary}'...")
        return pipeline_model_load(path_secondary)
    pipe_loaded = PipelineModel.read().load(path_primary)
    return pipe_loaded

# COMMAND ----------

from pyspark.sql import types as T

def transformer_feature_fillna(sdf_data, val_numeric=0, val_string=''):
    """
    Method to safely fill a dataframe acccording to type.
    One way to avoid the need for this operation is to use an 'imputer'  (check out notebook 1c!)
    """
    feature_set_config = sdf_data.columns
    schema_types = list(sdf_data.schema)
    
    # select columns that contain numeric data
    numeric_types = [T.IntegerType, T.BooleanType, T.DecimalType, T.DoubleType, T.FloatType]
    string_types = [T.StringType]
    cols_num = [c.name for c in schema_types if (type(c.dataType) in numeric_types) ]
    cols_str = [c.name for c in schema_types if (type(c.dataType) in string_types) ]
    sdf_filled = (sdf_data
        .fillna(val_numeric, cols_num)
        .fillna(val_string, cols_str)
    )
    return sdf_filled
