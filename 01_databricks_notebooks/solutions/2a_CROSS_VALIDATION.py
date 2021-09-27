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
# MAGIC 3. Navigate to this script (Repos -> ez2685@DOMAIN -> mlworkshop2021 -> 01_databricks_notebooks -> (solutions) -> `2a_CROSS_VALIDATION`)
# MAGIC 4. Clone the file (either `File` -> `Clone` or above with right-click)
# MAGIC    - Alternatively, add a repo (left side menu) for the [MLWorkshop2021](https://dsair@dev.azure.com/dsair/dsair_collaborator/_git/mlworkshop2021) that is stored in [Azure DevOps](https://dev.azure.com/dsair/dsair_collaborator/_git/mlworkshop2021) and jump into your own url..
# MAGIC 5. *Pat on back* or *high five*

# COMMAND ----------

# MAGIC %run ../utilities/WORKSHOP_CONSTANTS

# COMMAND ----------

# MAGIC %md
# MAGIC # Preprocessed Data

# COMMAND ----------

sdf_transformed = spark.read.format('delta').load(IHX_GOLD_TRANSFORMED)
sdf_transformed_test = spark.read.format('delta').load(IHX_GOLD_TRANSFORMED_TEST)
col_features = IHX_COL_VECTORIZED

# COMMAND ----------

# MAGIC %md
# MAGIC # Cross-fold Validation
# MAGIC quick eval
# MAGIC [cross-validator](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.CrossValidator.html)

# COMMAND ----------

# MAGIC %run ../utilities/MODEL_TOOLS

# COMMAND ----------

# MAGIC %run ../utilities/EVALUATOR_TOOLS

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator

method_test = "RF"
cf, grid = create_untrained_classifier(method_test, col_features, False)   # step one - get the regression classifier
evaluator = evaluator_obj_get('CG2D')   # a workshop function from "EVALUATOR_TOOLS"
cvModel = cf.fit(sdf_transformed_sample)
crossval = CrossValidator(estimator=cf, estimatorParamMaps=grid,
                          evaluator=evaluator, numFolds=3)  # use 3+ folds in practice

# Run cross-validation, and choose the best set of parameters.
sdf_transformed_sample = sdf_transformed.sample(IHX_TRAINING_SAMPLE_FRACTION)
cvModel = crossval.fit(sdf_transformed_sample)
fn_log(f"Average Cross-fold Metrics {cvModel.avgMetrics}")

# now perform prediction on our test set and try again
sdf_predict = cvModel.transform(sdf_transformed_test)
score_eval = evaluator.evaluate(sdf_predict)

str_title = f"Xfold-{method_test} CGD (2-decile): {round(score_eval, 3)} "
fn_log(str_title)
evaluator_performance_curve(sdf_predict, str_title)



# COMMAND ----------

str_title = f"Xfold-RF CGD (2-decile): {round(score_eval, 3)} "
fn_log(str_title)
evaluator_performance_curve(sdf_predict, str_title)


# COMMAND ----------

# MAGIC %md
# MAGIC # MLFlow tracking
# MAGIC Talk about adding parameters and other logging metrics to ml flow

# COMMAND ----------

# MAGIC %run ../utilities/MLFLOW_TOOLS

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator
import datetime as dt
import os
from tempfile import NamedTemporaryFile
import random

method_test = "RF"

experiment = databricks_mlflow_create(MLFLOW_EXPERIMENT)
# sdf_transformed_sample = sdf_transformed.sample(IHX_TRAINING_SAMPLE_FRACTION)
run_name = f"xval-{dt.datetime.now().strftime('%m%d-%H%S')}"

# start a new run with specific id, it will be closed at the end of this cell
mlflow.end_run()  # doesn't hurt, always close a prior run
with mlflow.start_run(run_name=run_name):
    print(mlflow.active_run().info)

    cf, grid = create_untrained_classifier(method_test, col_features, False)   # step one - get the regression classifier
    evaluator = evaluator_obj_get('CG2D')   # a workshop function from "EVALUATOR_TOOLS"
    cvModel = cf.fit(sdf_transformed_sample)

    # now perform prediction on our test set and try again
    sdf_predict = cvModel.transform(sdf_transformed_test)
    score_eval = evaluator.evaluate(sdf_predict)

    # write some various metrics, params, etc.
    mlflow.set_tag("s.release", "1.2.3")
    mlflow.log_param("column-choice", col_features)
    mlflow.log_param("sample_fraction", IHX_TRAINING_SAMPLE_FRACTION)
    mlflow.log_metric("user-provided", score_eval)
    mlflow.log_text(f"{dt.datetime.now()}: this would be a complete log to save...", 'log_text.txt')
    
    # you can also log different metric values with steps
    val_demo = 0
    for i in range(10):
        mlflow.log_metric("time-variant-metric", val_demo, i)
        val_demo += (random.random() * (1-val_demo))
    
    # plot a figure and log it to mlflow
    str_title = f"Xfold-{method_test} CGD (2-decile): {round(score_eval, 3)} "
    fn_log(str_title)
    fig = evaluator_performance_curve(sdf_predict, str_title)
    mlflow.log_figure(fig, "xfold-gcd.png")

    # you can also write a temp file...
    f = NamedTemporaryFile(mode='w+t', delete=False, suffix='.txt')
    f.write("This is a temp file, written in Databricks\n") 
    f.write(f"File written on {dt.datetime.now()} with score {score_eval}\n")
    f.close()
    mlflow.log_artifact(f.name, 'temp_text.txt')
    # clean up temp file
    os.unlink(f.name)



# COMMAND ----------

# MAGIC %md
# MAGIC Thanks for walking through this intermediate lesson on cross-fold validation and mlflow logging.   
# MAGIC 
# MAGIC When you're ready, head on to the next script `2b_GRID_EXPLORE` that includes more intensive grid search and model building in databricks.

# COMMAND ----------

# MAGIC %md
# MAGIC # Done with Features!
# MAGIC Still want more or have questions about more advanced topics?  Scroll back up to this script section labeled `extra_credit` to improve your pipeline and add imputing. 
