# Databricks notebook source
from pyspark.ml.tuning import TrainValidationSplit

def sampling_stratified(sdf, col_name, col_unique, dict_fractions, overall_sample=None):
    """
    Quick custom function for stratified sampling...; dict_fractions = {1:0.4, 0:0.6}
    """
    # this is the line that we want to execute, but as of 9/23 some security settings prevent it!
    # sdf_transformed_sampled = sdf.sampleBy(F.col(col_unique), fractions=dict_fractions, seed=0)
    sdf_working = sdf.withColumn('_salt', F.abs(F.hash(col_unique)) % 100)
    # loop through keys to compile our own dataframe
    for key_idx in dict_fractions:
        # apply a filter that will select/match the key and limit the number of samples
        salt_filter = int(dict_fractions[key_idx]*100)
        fn_log(f"[stratified_sample] Filter '{col_name}'=={key_idx}, limit: {salt_filter} %")
        sdf_working = sdf_working.filter((F.col(col_name) != key_idx) | (F.col('_salt') <= salt_filter))
    # if None overall sample, retturn as is
    if overall_sample is None:
        return sdf_working.drop(F.col('_salt'))      # all done? drop the random column we generated
    # subsample overall amount
    return sdf_working.drop(F.col('_salt')).sample(overall_sample)    # all done? drop the random column we generated
