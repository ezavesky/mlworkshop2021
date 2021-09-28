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
# MAGIC We prepared it for prior experiments, so let's start from transformed training and testing features directly.

# COMMAND ----------

sdf_transformed = spark.read.format('delta').load(IHX_GOLD_TRANSFORMED)
sdf_transformed_test = spark.read.format('delta').load(IHX_GOLD_TRANSFORMED_TEST)
col_features = IHX_COL_VECTORIZED
sdf_transformed_sample = sdf_transformed.sample(IHX_TRAINING_SAMPLE_FRACTION)


# COMMAND ----------

# MAGIC %md
# MAGIC # Grid Exploration
# MAGIC Grid exploration allows the evaluation of a large number of parameter settings for a model.  
# MAGIC 
# MAGIC First, we'll explore the total run time for grid-training of several (six) different parameter settigns.   Using our prior helper function `create_untrained_classifier` (to produce the variable `grid` with multiple parameter settings) and work to consolidate metric and parameter defintions, we'll track all of the grid exploration with mlflow runs.

# COMMAND ----------

# MAGIC %run ../utilities/MODEL_TOOLS

# COMMAND ----------

# MAGIC %run ../utilities/EVALUATOR_TOOLS

# COMMAND ----------

# MAGIC %run ../utilities/MLFLOW_TOOLS

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator
import datetime as dt

method_test = "LR"
cf, grid = create_untrained_classifier(method_test, col_features, True)   # step one - get the regression classifier
# fn_log(grid)  # what does the grid of options look like?

evaluator = evaluator_obj_get('CG2D')   # a workshop function from "EVALUATOR_TOOLS"
experiment = databricks_mlflow_create(MLFLOW_EXPERIMENT)
# sdf_transformed_sample = sdf_transformed.sample(IHX_TRAINING_SAMPLE_FRACTION)
run_name = f"grid-{method_test{dt.datetime.now().strftime('%m%d-%H%S')}"

# start a new run with specific id, it will be closed at the end of this cell
mlflow.end_run()  # doesn't hurt, always close a prior run
with mlflow.start_run(run_name=run_name):
    crossval = CrossValidator(estimator=cf, estimatorParamMaps=grid,
                              evaluator=evaluator, numFolds=2)  # use 3+ folds in practice

    # Run cross-validation, and choose the best set of parameters.
    cvModel = crossval.fit(sdf_transformed_sample)
    fn_log(f"Average Cross-fold Metrics {cvModel.avgMetrics}")

    # now perform prediction on our test set and try again
    sdf_predict = cvModel.transform(sdf_transformed_test)
    score_eval = evaluator.evaluate(sdf_predict)

    # mlflow.log_params(best_hyperparam)  # not available here!
    mlflow.spark.log_model(cvModel, "spark-model")
    mlflow.set_tag("search-type", "grid")

    # plot a figure and log it to mlflow
    str_title = f"Grid-{method_test} CGD (2-decile): {round(score_eval, 3)} (x-fold CGD: {round(cvModel.avgMetrics[0], 3)})"
    fn_log(str_title)
    fig = evaluator_performance_curve(sdf_predict, str_title)
    mlflow.log_figure(fig, "grid-gcd.png")




# COMMAND ----------

# MAGIC %md
# MAGIC ## Auto-logging and Grid Search
# MAGIC In the previous step of our experiments, we used cross validation to train many experiments.  Unfortuantely, two problems remain...
# MAGIC 1. We do not have the individual parameter settings of each model 
# MAGIC 2. All of the parameter settings were evaluated, but this is not time efficient.
# MAGIC 
# MAGIC Instead, we'll use another training library called [hyperopt](http://hyperopt.github.io/hyperopt/) to efficiently traverse a large parameter space for discovery of an optimal model. 
# MAGIC 
# MAGIC With the function definition below, we can also leverage mlflow model tracking.

# COMMAND ----------

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np

from pyspark.ml.tuning import CrossValidator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder

def paramdict_to_paramgrid(params, cf):
    """
        Convert a dictionary of values to a parameter map for training
    """
    # this is a little tricky, but we need to access our LR model
    param_grid = ParamGridBuilder()
    for k in params:
        param_grid.addGrid(getattr(cf, k), [params[k]])
    param_grid = param_grid.build()
    return param_grid

def objective_function(params):
    """
        An objective function that is used to evaluate different parameter settings with cross-validation
    """
    # create a grid with our hyperparameters
    cf, _ = create_untrained_classifier(method_test, col_features, False)
    evaluator = evaluator_obj_get('CG2D')
    fn_log(f"OBJECTIVE PARAMS: {params}")

    # evaluator = get_binary_evalutor(evaluator_metric)
    cv = CrossValidator(estimator=cf, evaluator=evaluator,
                        estimatorParamMaps=paramdict_to_paramgrid(params, cf),
                        numFolds=2,  parallelism=4, seed=42)
    cvModel = cv.fit(sdf_transformed_sample)

    # get our average primary metric across all three folds (make negative because we want to minimize)
    retVal = cvModel.avgMetrics[0]
    if evaluator.isLargerBetter():   # the calling function will always try to minimize, so flip sign if needed
        retVal = -retVal
    return {"loss": retVal, "params": params, "status": STATUS_OK}


# COMMAND ----------

# MAGIC %md
# MAGIC The code block below is our main hyperopt executor.  It uses the function above to evaluate one parameter setting at a time using a well-known cross-validator instance.

# COMMAND ----------

# start a new run with specific id, it will be closed at the end of this cell
mlflow.end_run()  # doesn't hurt, always close a prior run
max_evals = 4  # number of evals to explore...
run_name = f"mlopt-{method_test}-{dt.datetime.now().strftime('%m%d-%H%S')}"
with mlflow.start_run(run_name=run_name) as run:   # note that we tweaked the run_name to v2!
    trials = Trials()   # only works with one object!
    # mlflow.pyspark.ml.autolog(log_models=False)
    best_hyperparam = fmin(fn=objective_function,
                         space=param_grid_to_hyperopt(grid),
                         algo=tpe.suggest, 
                         max_evals=max_evals,   # how many parts of grid?
                         trials=trials,
                         rstate=np.random.RandomState(42)
                        )

    # this last chunk is to dereference the 'best' values; without it you may 
    # get an index reference (e.g. 0, 1, 2 instead of the actual values [0.01, 0.15, 0.1])
    best_hyperparam = None
    best_loss = 1e10
    list_performance = []
    for g in trials.trials:
        if best_hyperparam is None or best_loss > g['result']['loss']:
            best_hyperparam = g['result']['params']
            best_loss = g['result']['loss']
        list_performance.append([best_hyperparam, -best_loss])
        #for p in g['misc']['vals']:
        #    # mlflow.log_param(p, g['misc']['vals'][p][0])
        #    logger.info(p, g['misc']['vals'][p][0])

    # the "trials" object has all the goodies this time!
    #   https://github.com/hyperopt/hyperopt/wiki/FMin#13-the-trials-object
    print(f"MLflow Trials: {trials.trials}")
    print(f"MLflow hyperparam (after dereference): {best_hyperparam}")

    # evaluator = get_binary_evalutor(evaluator_metric)
    cf, _ = create_untrained_classifier(method_test, col_features, False)
    cv = CrossValidator(estimator=cf, evaluator=evaluator,
                        estimatorParamMaps=paramdict_to_paramgrid(best_hyperparam, cf),
                        numFolds=2,  parallelism=4, seed=42)
    cvModel = cv.fit(sdf_transformed_sample)
    mlflow.log_params(best_hyperparam)
    mlflow.spark.log_model(cvModel, "spark-model")
    mlflow.set_tag("search-type", "hyperopt")

    # now perform prediction on our test set and try again
    sdf_predict = cvModel.transform(sdf_transformed_test)
    score_eval = evaluator.evaluate(sdf_predict)

    # plot the performance figure
    str_title = f"Xfold-{method_test} CGD (2-decile): {round(score_eval, 3)} "
    fn_log(str_title)
    fig = evaluator_performance_curve(sdf_predict, str_title)

    # plot a figure and log it to mlflow
    mlflow.log_figure(fig, "xfold-gcd.png")
   

# COMMAND ----------

# MAGIC %md
# MAGIC Thanks for walking through this intermediate less on cross-fold validation and parameter search.  We explored grid-search and stochastically optimized parameter-search options for model fitting.  We also looked at best practice methods for integrating with [mlflow](https://mlflow.org/docs/latest/quickstart.html) to explore, log, and package models trained in databricks.
# MAGIC 
# MAGIC When you're ready, head on to the next script `1b_DATA_WRITE_EXAMPLES` that includes just a few examples of how to write your data in Spark.

# COMMAND ----------

# MAGIC %md
# MAGIC # Done with Grid Exploration and Logging!
# MAGIC Still want more or have questions about more logging model exploration?  Continue on to either notebook 2b or 2c! 
