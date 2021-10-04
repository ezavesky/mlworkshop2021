# Databricks notebook source
# MAGIC %md
# MAGIC # Sample Imbalance
# MAGIC Sometime, you may wonder why your classifier's doing okay but not great.  In our main workflow (workbook 1d), our primary classifier was evaluating with cumulative gain scores around 0.36-0.37 (the [Pinnacle leaderboard](https://pinnacle.SERVICE_SITE/usecase/id/f29e33307cb04960a6697d713b0053a8) had a baseline of 0.3816 and high performers around 0.43) respectively.
# MAGIC 
# MAGIC In our current configuration (that of sub-sampling) we may need to expose the classifier to more (or slightly more equal dsitributions of) examples from the training set.  

# COMMAND ----------

# MAGIC %run ../utilities/WORKSHOP_CONSTANTS

# COMMAND ----------

# MAGIC %run ../utilities/EVALUATOR_TOOLS

# COMMAND ----------

# MAGIC %run ../utilities/TRANSFORMER_TOOLS

# COMMAND ----------

# MAGIC %run ../utilities/MLFLOW_TOOLS

# COMMAND ----------

# MAGIC %run ../utilities/MODEL_TOOLS

# COMMAND ----------

# special command to engage in model tracking (full introduction in notebook `2a`)
experiment = databricks_mlflow_create(MLFLOW_EXPERIMENT)

# load trained model for inspection
pipe_model_trained = pipeline_model_load(SCRATCH_IHX_TRANSFORMER_MODEL_PATH, IHX_TRANSFORMER_MODEL_PATH)
col_features = pipe_model_trained.stages[-1].getOutputCol()

# load transformed data
sdf_transformed = model_load_data(SCRATCH_IHX_GOLD_TRANSFORMED, IHX_GOLD_TRANSFORMED)
sdf_transformed_test = model_load_data(SCRATCH_IHX_GOLD_TRANSFORMED_TEST, IHX_GOLD_TRANSFORMED_TEST)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Overfitting
# MAGIC One challenge that we need to cognizant of is [overfitting](https://en.wikipedia.org/wiki/Overfitting), where the classifier tries to get all samples correct but in doing so (particularly in more statistiacl learners like Bayesian classifiers and linear egressors), it makes for a worse classifier on new data.  

# COMMAND ----------

# reduce training set from 429k to about 10k for speed during the workshop
sdf_train = sdf_transformed.sample(IHX_TRAINING_SAMPLE_FRACTION)
evaluator = evaluator_obj_get('CG2D')   # a workshop function from "EVALUATOR_TOOLS"

# Fit/train the model
cf, grid = create_untrained_classifier("RF", col_features)
cfModel = cf.fit(sdf_train)

# evaluate the model and show just a few columns about probabilistic and final decision values
sdf_predict = cfModel.transform(sdf_transformed_test).drop(IHX_COL_NORMALIZED)
display(sdf_predict)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample Imbalance
# MAGIC To compensate, statified sampling (with pyspark's [sampleBy](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.sampleBy.html?highlight=stratified) function) will allow each class to have close-to-equal representation.  Genrally, clasifiers can handle large disparities, but for those that fail, a rate more than a 8-to-1 should be avoided. 
# MAGIC 
# MAGIC At the same time, we'll also use the spark "all-in-one" class, [TrainValidationSplit](https://spark.apache.org/docs/latest/ml-tuning.html#train-validation-split) for training (on a fraction of the training data) and validating on the rest (the hold-out data).  

# COMMAND ----------

# count the number of labels and convert to easier pandas dataframe
df_class_counts = (sdf_transformed
    .groupBy(F.col(IHX_COL_LABEL))
    .agg(F.count(IHX_COL_LABEL).alias('count'))
    .toPandas()
)
display(df_class_counts)

# okay, get the smallest count from both classes and use that
min_count = df_class_counts['count'].min()

# now, let's compute a sampling ratio for stratified sampling
max_imbalance = 5.0
max_count = min_count * max_imbalance
dict_fractions = df_class_counts.set_index(IHX_COL_LABEL).to_dict(orient='index')
fn_log(dict_fractions)
dict_fractions = {class_key: (min(max_count, dict_fractions[class_key]['count'])/dict_fractions[class_key]['count']) 
                  for class_key in dict_fractions}
fn_log(f"Min Class Count {min_count}, Max imbalance: {max_imbalance}, Sample ratio: {dict_fractions}")

# COMMAND ----------

# MAGIC %run ../utilities/SAMPLING_TOOLS

# COMMAND ----------

# apply our stratified sampled
sdf_transformed_stratified = sampling_stratified(sdf_transformed, IHX_COL_LABEL, IHX_COL_INDEX, 
                                              dict_fractions, IHX_TRAINING_SAMPLE_FRACTION)
# display(sdf_transformed_sampled)

# and finally specify some hold-out for valiation
ratio_validation = 0.1
tvs = TrainValidationSplit(estimator=cf, trainRatio=(1 - ratio_validation),
                           estimatorParamMaps=grid, evaluator=evaluator,
                           parallelism=1, seed=42)
cfModel = tvs.fit(sdf_transformed_stratified)
num_train = sdf_transformed_stratified.count()

# now perform prediction on our test set and try again
sdf_predict = cfModel.transform(sdf_transformed_test)
score_eval = evaluator.evaluate(sdf_predict)

str_title = f"Stratified CGD (2-decile, {num_train} samples): {score_eval}\n(validation CGD: {cfModel.validationMetrics})"
fn_log(str_title)
evaluator_performance_curve(sdf_predict, str_title)
