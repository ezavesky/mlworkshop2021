# Databricks notebook source
# MAGIC %md
# MAGIC # Part 2 - MLWorkshop2021
# MAGIC This notebook is part of the [2021 ML Workshop](https://INFO_SITE/cdo/events/internal-events/90dea45e-1454-11ec-8dca-7d45a6b8dd2a) (sponsored by the Software Symposium)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Clone This Demo Notebook
# MAGIC 1. Head to http://FORWARD_SITE/cdo-databricks  (don't worry, it's a fowarding DNS to the main CDO PROD(uction) instance)
# MAGIC 2. Log-in via your AT&T ID and DOMAIN password
# MAGIC 3. Navigate to this script (Repos -> ez2685@DOMAIN -> mlworkshop2021 -> 01_databricks_notebooks -> (solutions) -> `2c_EXPERT_PARITY`)
# MAGIC 4. Clone the file (either `File` -> `Clone` or above with right-click)
# MAGIC    - Alternatively, add a repo (left side menu) for the [MLWorkshop2021](https://dsair@dev.azure.com/dsair/dsair_collaborator/_git/mlworkshop2021) that is stored in [Azure DevOps](https://dev.azure.com/dsair/dsair_collaborator/_git/mlworkshop2021) and jump into your own url..
# MAGIC 5. *Pat on back* or *high five*

# COMMAND ----------

# MAGIC %run ../utilities/WORKSHOP_CONSTANTS

# COMMAND ----------

# MAGIC %md
# MAGIC # Preprocessed Data
# MAGIC We prepared it for prior experiments, so let's start from transformed training and testing features directly.

# COMMAND ----------

sdf_transformed = spark.read.format('delta').load(IHX_GOLD_TRANSFORMED)
sdf_transformed_test = spark.read.format('delta').load(IHX_GOLD_TRANSFORMED_TEST)
col_features = IHX_COL_VECTORIZED
sdf_transformed_sample = sdf_transformed.sample(IHX_TRAINING_SAMPLE_FRACTION)


# COMMAND ----------

# MAGIC %run ../utilities/MODEL_TOOLS

# COMMAND ----------

# MAGIC %run ../utilities/EVALUATOR_TOOLS

# COMMAND ----------

# MAGIC %run ../utilities/MLFLOW_TOOLS

# COMMAND ----------

# MAGIC %md
# MAGIC # Expert Parity
# MAGIC We learned that [lightgbm](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html#lightgbm.train) was the preferred modeling system, so let's go for that ourselves!  The function below skips any kind of parameter search or cross-fold validation because we directly use the parameter settings that were identified as best from the talk.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pandas Features
# MAGIC In this case, we'll be using a model interface that works on the [scikit-learn](scikit-learn.org/) pattern.  That means that we'll have to export our spark-based data into the [Pandas dataframe](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) format with the function below.

# COMMAND ----------

import pyspark.sql.functions as F
import pyspark.sql.types as T
from lightgbm import LGBMClassifier, LGBMRanker
import datetime as dt
import numpy as np
import pandas as pd

def pandas_features(sdf_source, col_features, list_columns=None):
    """
    Flatten our spark dataframe into pandas into X (samples) and Y (labels)
    """    
    df_source = (
        sdf_source.select(F.col(IHX_COL_INDEX), F.col(col_features), F.col(IHX_COL_LABEL))
        .toPandas()
        .set_index(IHX_COL_INDEX, drop=True)
    )
    df_Y = df_source[IHX_COL_LABEL]
    df_X = pd.DataFrame(np.vstack(df_source[col_features]), 
                        # columns=list(df_source.toPandas()['FEATURE_COLS'][0]),
                        index=df_source.index)
    if list_columns is not None:
        df_X.columns = list_columns

    fn_log(f"Total samples: {len(df_X)}, total features: {len(df_X.columns)}")
    return df_X, df_Y


# COMMAND ----------

# MAGIC %md
# MAGIC ## Train 'lightgbm'
# MAGIC Training and evaluation is pretty straight forward.  For easier reuse of our existing metrics (in spark), we'll transform the resutls from this prediction back after running it through lightgbm.

# COMMAND ----------

from lightgbm import LGBMClassifier, LGBMRanker


# start local experiment
experiment = databricks_mlflow_create(MLFLOW_EXPERIMENT)
method_test = "LGBM"
max_evals = 4  # number of evals to explore...
evaluator = evaluator_obj_get('CG2D')   # a workshop function from "EVALUATOR_TOOLS"

# start a new run with specific id, it will be closed at the end of this cell
mlflow.end_run()  # doesn't hurt, always close a prior run
run_name = f"expert-{method_test}-{dt.datetime.now().strftime('%m%d-%H%S')}"
with mlflow.start_run(run_name=run_name) as run:
    # proceed to train
    df_X, df_Y = pandas_features(sdf_transformed, col_features)

    # other parameters to try?...
    # https://testlightgbm.readthedocs.io/en/latest/Parameters.html#parameter-format
    # learning_rate = 0.1
    # num_leaves = 255
    # num_trees = 500
    # num_threads = 16
    # min_data_in_leaf = 0
    # min_sum_hessian_in_leaf = 100
    # num_round = 10
    param_learn = {"boosting_type": 'gbdt', "num_leaves": 31, "max_depth": 14, 
                   "learning_rate": 0.01, "n_estimators": 650, 
                   "reg_alpha": 0.0, "reg_lambda": 0.0}
        
    cf = LGBMClassifier(**param_learn)
    cvModel = cf.fit(df_X, df_Y, eval_metric='auc')
    mlflow.sklearn.log_model(cvModel, "core-model")
    mlflow.log_params(param_learn)
    mlflow.set_tag("search-type", "direct")

    # proceed to predict...
    df_X, df_Y = pandas_features(sdf_transformed_test, col_features)
    nd_predict = cf.predict_proba(df_X)

    # convert from pandas prediction to spark
    df_predict = pd.DataFrame([[nd_predict[i].tolist(), df_Y.iloc[i]] for i in range(len(df_Y))])
    method_test = "LGBM"
    schema = T.StructType([
       T.StructField(IHX_COL_PREDICT_BASE.format(base="prob"), T.ArrayType(T.DoubleType()), True),
       T.StructField(IHX_COL_LABEL, T.IntegerType(), True)])
    sdf_predict = spark.createDataFrame(df_predict, schema)

    # get evaluator and evaluate
    score_eval = evaluator.evaluate(sdf_predict)
    mlflow.log_metric(evaluator.getMetricName(), score_eval)

    # plot a figure and log it to mlflow
    str_title = f"Grid-{method_test} CGD (2-decile): {round(score_eval, 3)}"
    fn_log(str_title)
    fig = evaluator_performance_curve(sdf_predict, str_title)
    mlflow.log_figure(fig, "grid-gcd.png")




# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Looking for Results
# MAGIC We used our mlflow structure to save the performance and model directly, so when we're ready to go, it should be directly accessible. Go back to your tracking notebook `MLWorkshop2021` (under `Workspace` -> `ID@DOMAIN` -> `MLWorkshop2021`) to explore the saved model and perforamnce.

# COMMAND ----------

# MAGIC %md
# MAGIC Thanks for walking through this final exercise of training our "best of" classifiers.  We logged another model but also created a model that matches the parameter settings specified by our expert with [mlflow](https://mlflow.org/docs/latest/quickstart.html).
# MAGIC 
# MAGIC When you're ready, head on to the next script `2d_DEPLOY_FOR_DASHBOARD` that includes some final data and dashboard anaysis in Spark.

# COMMAND ----------

# MAGIC %md
# MAGIC # Done with Grid Exploration and Logging!
# MAGIC Still want more or have questions about more logging model exploration?  Continue on to either notebook 2b or 2c! 
