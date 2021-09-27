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
# MAGIC 3. Navigate to this script (Repos -> ez2685@DOMAIN -> mlworkshop2021 -> 01_databricks_notebooks -> (solutions) -> `2b_GRID_EXPLORE`)
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
# MAGIC # Grid Exploration
# MAGIC grid exploration

# COMMAND ----------

# MAGIC %run ../utilities/MODEL_TOOLS

# COMMAND ----------

# MAGIC %run ../utilities/EVALUATOR_TOOLS

# COMMAND ----------

# MAGIC %run ../utilities/MLFLOW_TOOLS

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator

cf, grid = create_untrained_classifier("RF", col_features, True)   # step one - get the regression classifier

experiment = databricks_mlflow_create(MLFLOW_EXPERIMENT)
# sdf_transformed_sample = sdf_transformed.sample(IHX_TRAINING_SAMPLE_FRACTION)
run_name = f"xval-{dt.datetime.now().strftime('%m%d-%H%S')}"

# start a new run with specific id, it will be closed at the end of this cell
mlflow.end_run()  # doesn't hurt, always close a prior run
with mlflow.start_run(run_name=run_name):

    crossval = CrossValidator(estimator=cf, estimatorParamMaps=grid,
                              evaluator=evaluator, numFolds=3)  # use 3+ folds in practice

    # Run cross-validation, and choose the best set of parameters.
    cvModel = crossval.fit(sdf_transformed_sample)
    fn_log(f"Average Cross-fold Metrics {cvModel.avgMetrics}")

    # now perform prediction on our test set and try again
    sdf_predict = cvModel.transform(sdf_transformed_test)
    score_eval = evaluator.evaluate(sdf_predict)

    # plot a figure and log it to mlflow
    str_title = f"Grid-{method_test} CGD (2-decile): {round(score_eval, 3)} (x-fold CGD: {round(cvModel.avgMetrics[0], 3)})"
    fn_log(str_title)
    fig = evaluator_performance_curve(sdf_predict, str_title)
    mlflow.log_figure(fig, "grid-gcd.png")



# COMMAND ----------

# MAGIC %md
# MAGIC # MLFlow tracking
# MAGIC Talk about adding parameters and other logging metrics to ml flow

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
# MAGIC ## Auto-logging and Grid Search
# MAGIC 
# MAGIC * In the last step of our experiments, we will use a feature called `autolog` in mlflow to record the model, parameters, and metrics.
# MAGIC * Additionally, to iterate through a large search space (e.g. grid search), we'll use another libary called [hyperopt](http://hyperopt.github.io/hyperopt/)

# COMMAND ----------

import pickle
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np

from pyspark.ml.tuning import CrossValidator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder

def objective_function(params):
    # create a grid with our hyperparameters
    pipeline, model_labels = get_untrained_lr(df_partial, 
        regParam=params["regParam"], maxIter=int(params["maxIter"]),
        elasticNetParam=params["elasticNetParam"])   # get the model pipeline again
    multi_eval = get_evaluator()
    print(f"OBJECTIVE PARAMS (RF): {params}, Metric: {multi_eval}")

    # this is a little tricky, but we need to access our LR model
    lr = pipeline.getStages()[-2]
    grid = (ParamGridBuilder()
            .addGrid(lr.regParam, [params["regParam"]])
            .addGrid(lr.maxIter, [int(params["maxIter"])])
            .addGrid(lr.elasticNetParam, [params["elasticNetParam"]])
            .build())

    # cross validate the set of hyperparameters
    multi_eval.setMetricName("accuracy")
    # evaluator = get_binary_evalutor(evaluator_metric)
    cv = CrossValidator(estimator=pipeline, evaluator=multi_eval,
                        estimatorParamMaps=grid,
                        numFolds=2, parallelism=4, seed=42)
    cvModel = cv.fit(df)

    # get our average primary metric across all three folds (make negative because we want to minimize)
    retVal = cvModel.avgMetrics[0]
    if multi_eval.isLargerBetter():   # the calling function will always try to minimize, so flip sign if needed
        retVal = -retVal
    return {"loss": retVal, "params": params, "status": STATUS_OK}


# http://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.LogisticRegression.html
#   update 7/20 from JD, shrink regParam search range (close to zero) -- https://www.slideshare.net/dbtsai/2014-0620-mlor-36132297
search_space = {
    "elasticNetParam": hp.choice("elasticNetParam", [0.3, 0.5, 0.7]),
    "regParam": hp.choice("regParam", [0.0, 0.05, 0.001]),
    "maxIter": hp.uniform('maxIter', 10, 100)
}


experiment = databricks_mlflow_create("WALK07-mlflow-imdb")

with mlflow.start_run(run_name=f"v2 @ {dt.datetime.now()}") as run:   # note that we tweaked the run_name to v2!
    df_partial = df.sample(0.15)   # for expensive grid training, try smaller sample
    trials = Trials()   # only works with one object!
    # mlflow.pyspark.ml.autolog(log_models=False)
    best_hyperparam = fmin(fn=objective_function,
                         space=search_space,
                         algo=tpe.suggest, 
                         max_evals=4,   # how many parts of grid?
                         trials=trials,
                         rstate=np.random.RandomState(42)
                        )

    # this last chunk is to dereference the 'best' values; without it you may 
    # get an index reference (e.g. 0, 1, 2 instead of the actual values [0.01, 0.15, 0.1])
    best_hyperparam = None
    best_loss = 1e10
    for g in trials.trials:
        if best_hyperparam is None or best_loss > g['result']['loss']:
            best_hyperparam = g['result']['params']
            best_loss = g['result']['loss']
        #for p in g['misc']['vals']:
        #    # mlflow.log_param(p, g['misc']['vals'][p][0])
        #    logger.info(p, g['misc']['vals'][p][0])

    # the "trials" object has all the goodies this time!
    #   https://github.com/hyperopt/hyperopt/wiki/FMin#13-the-trials-object
    print(f"MLflow Trials: {trials.trials}")
    print(f"MLflow hyperparam (after dereference): {best_hyperparam}")

    pipeline, model_labels = get_untrained_lr(df, **best_hyperparam)   # get the model pipeline again
    model = evaluate_and_log_model(pipeline, df, best_hyperparam)   # call function to retrain


    
    

# COMMAND ----------

# MAGIC %md
# MAGIC Thanks for walking through this intermediate less on cross-fold validation and parameter search.   
# MAGIC 
# MAGIC When you're ready, head on to the next script `1b_DATA_WRITE_EXAMPLES` that includes just a few examples of how to write your data in Spark.

# COMMAND ----------

# MAGIC %md
# MAGIC # Done with Features!
# MAGIC Still want more or have questions about more advanced topics?  Scroll back up to this script section labeled `extra_credit` to improve your pipeline and add imputing. 
