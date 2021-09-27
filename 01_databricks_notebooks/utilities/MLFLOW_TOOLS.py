# Databricks notebook source
# MLFlow Template
# Allows quick create or erase of mlflow experiments in your home directory.
# 
# * `databricks_mlflow_create(notebook_name)`
#   * create a new notebook name (both on DBFS and in your workspace) as specified by the parameter
# * `cross_fold_predictions(dataframe, untrained_model, hash_col, num_folds=3)`
#   * given a `dataframe` and `untrained model`, use the column `hash_col` to generate `num_folds` different partitions for trainig and evaluation using a "hold out" set
#   
# **NOTE** You must be working on a cluster with an "ML" image or have installed mlflow (risky to do it yourself!)

# COMMAND ----------

# MAGIC %run ../utilities/WORKSHOP_CONSTANTS

# COMMAND ----------

import mlflow

def databricks_mlflow_create(notebook_name):
    name_experiment = f"{NOTEBOOK_BASE}/{notebook_name}"
    #  1. create the experiment once via the notebook, like: 
    try:
        experiment_id = mlflow.create_experiment(
            name_experiment, f"dbfs:{EXPERIMENT_BASE}/{notebook_name}")
    except Exception as e:
        print(f"Failed to create experiment, did it exist?: {e}")
    #  2. then get the experiment via this command: 
    experiment = mlflow.get_experiment_by_name(name_experiment)
    #  3. finally set the experiment via this command: 
    mlflow.set_experiment(experiment.name)
    return experiment
  
def databricks_mlflow_delete(notebook_name):
    # for now, we interpret this request to delete generated files...
    name_experiment = f"{NOTEBOOK_BASE}/{notebook_name}"
    experiment = mlflow.get_experiment_by_name(name_experiment)
    if experiment is not None:
        mlflow.delete_experiment(experiment.experiment_id)
        dbutils.fs.rm(f"dbfs:/{EXPERIMENT_BASE}/{notebook_name}", True)


# COMMAND ----------

def cross_fold_predictions(dataframe, untrained_model, hash_col, num_folds=3):
    output_df = None

    # logger.info(f"num_folds: {num_folds}")
    for fold in range(num_folds):
        # split data into training and testing dataframe
        training_data = dataframe.filter(f"pmod(hash({hash_col}), {num_folds}) != {fold}")
        testing_data = dataframe.filter(f"pmod(hash({hash_col}), {num_folds}) == {fold}")

        # get trained model
        model = untrained_model.fit(training_data)

        # make predictions
        predictions_df = model.transform(testing_data)

        # append to output dataset
        if output_df is None:
            output_df = predictions_df
        else:
            output_df = output_df.union(predictions_df)

    return output_df

