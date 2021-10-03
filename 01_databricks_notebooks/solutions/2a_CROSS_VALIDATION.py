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
# MAGIC We prepared it for prior experiments, so let's start from transformed training and testing features directly.

# COMMAND ----------

sdf_transformed = spark.read.format('delta').load(IHX_GOLD_TRANSFORMED)
sdf_transformed_test = spark.read.format('delta').load(IHX_GOLD_TRANSFORMED_TEST)
col_features = IHX_COL_VECTORIZED
sdf_transformed_sample = sdf_transformed.sample(IHX_TRAINING_SAMPLE_FRACTION)


# COMMAND ----------

# MAGIC %md
# MAGIC # Cross-fold Validation
# MAGIC Cross validation allows you to quickly evaluate a model's parameter settings across subsets of training and testing data.  Spark's [cross-validator class](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.CrossValidator.html) automates the partitionining and averaging of a single training set with a specific evaluator (or metric).
# MAGIC 
# MAGIC In the example below, we also compare a three-fold evaluation on the training set versus direct evaluation on the testing set.  We can see that the cross-fold score isn't quite as accurate as the direct evaluator, but it does allow a faster evaluation of a model.

# COMMAND ----------

# MAGIC %run ../utilities/MODEL_TOOLS

# COMMAND ----------

# MAGIC %run ../utilities/EVALUATOR_TOOLS

# COMMAND ----------

evaluator_performance_curve(sdf_predict, str_title)

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator

method_test = "RF"
cf, grid = create_untrained_classifier(method_test, col_features, False)   # step one - get the regression classifier
evaluator = evaluator_obj_get('CG2D')   # a workshop function from "EVALUATOR_TOOLS"
cvModel = cf.fit(sdf_transformed_sample)
crossval = CrossValidator(estimator=cf, estimatorParamMaps=grid,
                          evaluator=evaluator, numFolds=3)  # use 3+ folds in practice

# Run cross-validation, and choose the best set of parameters.
cvModel = crossval.fit(sdf_transformed_sample)
fn_log(f"Average Cross-fold Metrics {cvModel.avgMetrics}")

# now perform prediction on our test set and try again
sdf_predict = cvModel.transform(sdf_transformed_test)
score_eval = evaluator.evaluate(sdf_predict)

str_title = f"Xfold-{method_test} CGD (2-decile): {round(score_eval, 3)} "
fn_log(str_title)
evaluator_performance_curve(sdf_predict, str_title)



# COMMAND ----------

# MAGIC %md
# MAGIC # MLFlow tracking
# MAGIC [MLflow](https://mlflow.org/docs/latest/index.html) is a powerful open-source library that provides utlity functions to wrap performance [logging](https://mlflow.org/docs/latest/tracking.html#logging-data-to-runs) and [packaging](https://mlflow.org/docs/latest/models.html#storage-format).  In the journey of model development to publication and MLoperations, it is AT&T's recommended format for cross-environment portability.  Some advantages over other wrapping/exporting of libraries is portability into [different environments](https://mlflow.org/docs/latest/tracking.html#performance-tracking-with-metrics) and generic wrapping from different languages like [R](https://mlflow.org/docs/latest/R-api.html), [java](https://mlflow.org/docs/latest/java_api/index.html) and [python](https://mlflow.org/docs/latest/python_api/index.html).
# MAGIC 
# MAGIC If you look carefully, you may have noticed that noticed the "experiments" section of a page [automatically getting populated](https://mlflow.org/docs/latest/tracking.html#automatic-logging) with each training.  This is a unique integration within Databricks, but we can go further to consolidate the destination and training of these models through code below.  
# MAGIC 
# MAGIC MLFlow also uses the following hierarchy...
# MAGIC * **Project** -> _a format for packaging data science code in a reusable and reproducible way_
# MAGIC     * **Experiment** -> _multiple runs (code executions) under experiments with different settings_
# MAGIC         * **Run** -> _a single execution of machine learning code_
# MAGIC             * **Model** - _a standard format for packaging machine learning models and code_
# MAGIC 
# MAGIC ... but we'll focus on those components below a run in the workshop.

# COMMAND ----------

# MAGIC %run ../utilities/MLFLOW_TOOLS

# COMMAND ----------

# MAGIC %md
# MAGIC ### Destination specification
# MAGIC The helper functions above (like `databricks_mlflow_create`) allow specification of workspace and drive destinations with the creation of an [**experiment**](https://mlflow.org/docs/latest/tracking.html#id9).
# MAGIC 
# MAGIC The code below shows how to programatically start and name a [**run**](https://mlflow.org/docs/latest/tracking.html#launching-multiple-runs-in-one-program).

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
mlflow.start_run(run_name=run_name)
fn_log(mlflow.active_run().info)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Custom metrics
# MAGIC The code below demonstrates how new models are associated with the open run.

# COMMAND ----------

# retrieve a classifier, train it as we did above
cf, grid = create_untrained_classifier(method_test, col_features, False)   # step one - get the regression classifier
evaluator = evaluator_obj_get('CG2D')   # a workshop function from "EVALUATOR_TOOLS"
cvModel = cf.fit(sdf_transformed_sample)

# now perform prediction on our test set and try again
sdf_predict = cvModel.transform(sdf_transformed_test)
score_eval = evaluator.evaluate(sdf_predict)

# plot the performance figure
str_title = f"Xfold-{method_test} CGD (2-decile): {round(score_eval, 3)} "
fn_log(str_title)
fig = evaluator_performance_curve(sdf_predict, str_title)

# you can also log different metric values with steps
val_demo = 0
for i in range(10):
    mlflow.log_metric("time-variant-metric", val_demo, i)
    val_demo += (random.random() * (1-val_demo))

# create a temp file that is also logged...
f = NamedTemporaryFile(mode='w+t', delete=False, suffix='.txt')
f.write("This is a temp file, written in Databricks\n") 
f.write(f"File written on {dt.datetime.now()} with score {score_eval}\n")
f.close()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Challenge 5
# MAGIC Let's get a quick hands-on for adding parameters, tags, figures and artifacts to a specific mlflow model.
# MAGIC 
# MAGIC Using the hints in this notebook, quickly build a method to accomplish these steps.
# MAGIC 1. Create a [custom tag](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_tag) (of your choosing) for the run model.
# MAGIC 2. Use [parameter logging](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_param) to record the value of chosen columns (`col_features`) and the sampel fraction (`IHX_TRAINING_SAMPLE_FRACTION`)
# MAGIC 3. Log specific [metrics](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_metric) like the evaluated metric stored in `score_eval`  on the testing dataset.
# MAGIC 4. Log the [generated graphic/metric](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_figure) that was generated above and stored in the variable `fig`.

# COMMAND ----------

#### CHALLENGE #### 

# write some various metrics, params, etc.
# mlflow.set_tag("???", "??")
# mlflow.log_param("??", ???)
# mlflow.log_metric("???", ???)
mlflow.log_text(f"{dt.datetime.now()}: this would be a complete log to save...", 'log_text.txt')

# plot a figure and log it to mlflow
# the variable `fig` is provided above, how can you write it to a saved artifact
# mlflow.log_figure(???)

#### CHALLENGE #### 

# you can also write a temp file...
mlflow.log_artifact(f.name, 'temp_text.txt')

# and hte model artifact
mlflow.spark.log_model(cvModel, "spark-model")



# COMMAND ----------

#### SOLUTION #### 

# write some various metrics, params, etc.
mlflow.set_tag("s.release", "1.2.3")
mlflow.log_param("column-choice", col_features)
mlflow.log_param("sample_fraction", IHX_TRAINING_SAMPLE_FRACTION)
mlflow.log_metric("user-provided", score_eval)
mlflow.log_text(f"{dt.datetime.now()}: this would be a complete log to save...", 'log_text.txt')

# plot a figure and log it to mlflow
mlflow.log_figure(fig, "xfold-gcd.png")

#### SOLUTION #### 

# you can also write a temp file...
mlflow.log_artifact(f.name, 'temp_text.txt')

# and hte model artifact
mlflow.spark.log_model(cvModel, "spark-model")



# COMMAND ----------

# MAGIC %md
# MAGIC Finally, remove the temp file created and be sure to end/close the run.

# COMMAND ----------

# clean up temp file
os.unlink(f.name)
# close the run
mlflow.end_run()  # doesn't hurt, always close a prior run

# COMMAND ----------

# MAGIC %md
# MAGIC Thanks for walking through this intermediate lesson on cross-fold validation and mlflow logging.   
# MAGIC 
# MAGIC When you're ready, head on to the next script `2b_GRID_EXPLORE` that includes more intensive grid search and model building in databricks.

# COMMAND ----------

# MAGIC %md
# MAGIC # Done with Cross-validation and Logging!
# MAGIC Still want more or have questions about more advanced topics?  There's more of this topic in the next workbook, so please proceed to notebook 2b!
