# Databricks notebook source
# MAGIC %md
# MAGIC # Part 1 - MLWorkshop2021
# MAGIC This notebook is part of the [2021 ML Workshop](https://INFO_SITE/cdo/events/internal-events/90dea45e-1454-11ec-8dca-7d45a6b8dd2a) (sponsored by the Software Symposium)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Clone This Demo Notebook
# MAGIC 1. Head to http://FORWARD_SITE/cdo-databricks  (don't worry, it's a fowarding DNS to the main CDO PROD(uction) instance)
# MAGIC 2. Log-in via your AT&T ID and DOMAIN password
# MAGIC 3. Navigate to this script (Repos -> ez2685@DOMAIN -> mlworkshop2021 -> 01_databricks_notebooks -> (solutions) -> `1d_MODELING_AND_METRICS`)
# MAGIC 4. Clone the file (either `File` -> `Clone` or above with right-click)
# MAGIC    - Alternatively, add a repo (left side menu) for the [MLWorkshop2021](https://dsair@dev.azure.com/dsair/dsair_collaborator/_git/mlworkshop2021) that is stored in [Azure DevOps](https://dev.azure.com/dsair/dsair_collaborator/_git/mlworkshop2021) and jump into your own url..
# MAGIC 5. *Pat on back* or *high five*

# COMMAND ----------

# MAGIC %run ../utilities/WORKSHOP_CONSTANTS

# COMMAND ----------

# MAGIC %md
# MAGIC # Reviewing Serialized Pipelines
# MAGIC In our last notebook we created a few pipelines that will convert from raw input dataframes into vecctorized feature rows.  Let's make sure these transformers are portable by loading and testing a quick fit operation.

# COMMAND ----------

# MAGIC %run ../utilities/TRANSFORMER_TOOLS

# COMMAND ----------

# load trained model for inspection
pipe_model_trained = pipeline_model_load(SCRATCH_IHX_TRANSFORMER_MODEL_PATH, IHX_TRANSFORMER_MODEL_PATH)

# print the new pipeline
fn_log(pipe_model_trained)
fn_log(pipe_model_trained.stages)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reading Transformed Data
# MAGIC Good separation of ETL and learning allows us to proceed directly with the ETL'd features from here on.  If you're curious about what ETL is or how we processed our raw features (bronze) to our transformed features (gold), flip back to notebook `1c`.  Specifically, even though we may train different ML models, they will all reuse the same preprocessed features.  This will reduce our processing time and make sure that the different models have the same starting point.

# COMMAND ----------

# MAGIC %run ../utilities/MODEL_TOOLS

# COMMAND ----------

sdf_transformed = model_load_data(SCRATCH_IHX_GOLD_TRANSFORMED, IHX_GOLD_TRANSFORMED)
sdf_transformed_test = model_load_data(SCRATCH_IHX_GOLD_TRANSFORMED_TEST, IHX_GOLD_TRANSFORMED_TEST)

# COMMAND ----------

# MAGIC %md
# MAGIC # Classifier Construction
# MAGIC At this point, a pipeline to transform features has been constructed and evaluated on our raw data.  Now, the task is to train and evaluate a classifier.  One of the bigger shifts to using a framework like Databricks is that it is spark driven and can show advantages (and limitations) with its distribtued environment.
# MAGIC 
# MAGIC Although there are others, we'll first explort the set of [classifiers](https://spark.apache.org/docs/latest/ml-classification-regression.html) natively available in spark.

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier

# get output feature as our column for input features
col_features = pipe_model_trained.stages[-1].getOutputCol()

#  get a raw clasifier; sample for single random forest classifier
cf = RandomForestClassifier(featuresCol=col_features, labelCol=IHX_COL_LABEL,
                    predictionCol=IHX_COL_PREDICT_BASE.format(base="int"),
                    probabilityCol=IHX_COL_PREDICT_BASE.format(base="prob"),
                    rawPredictionCol=IHX_COL_PREDICT_BASE.format(base="raw"))
cf.setNumTrees(100)
cf.setMaxDepth(10)

# we've colleted these function in this helper file... '../utilities/MODEL_TOOLS'
#    def create_untrained_classifier(classifier="RF", col_features=col_features):


# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Logging
# MAGIC We'll explore this in notebook `2a`, but for now, know that there is a method to help log metrics and save models automatically.  We're setting the stage for future exploration of these results with the helper script and command below.

# COMMAND ----------

# MAGIC %run ../utilities/MLFLOW_TOOLS

# COMMAND ----------

# special command to engage in model tracking (full introduction in notebook `2a`)
experiment = databricks_mlflow_create(MLFLOW_EXPERIMENT)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Training and Evaluation
# MAGIC In these classifiers, the basic workflow is parameter setting, training (executed with `fit()`) and evaluation (executed with `transform()`).  Let's see how it works with two common classifiers, [Random Forests](https://spark.apache.org/docs/latest/ml-classification-regression.html#random-forest-classifier) and [Logistic Regression](https://spark.apache.org/docs/latest/ml-classification-regression.html#multinomial-logistic-regression).  
# MAGIC 
# MAGIC The next topic of concern is data sampling.
# MAGIC 1. Normally you'd have to split your training data into a training set and a testing (or holdout) set.  Typically, you don't have a lot of fully annotated data, so this practice creates a safe/unseen set of data that can make sure your training and evaluation is as close to real world conditions as possible. To keep things similar to real-world data, you should make the sample choice random.
# MAGIC 2. For the workshop, we're going to shrink our training sample count to keep performance snappy. So, while we have over 429k training samples (and 118 features), we're going to reduce that to an even 50k samples. 
# MAGIC     * **NOTE**: In a spark environment this is totally unnecessary and generally you'd never short-change yourself with samples!  However, to "keep it real" we are evaluating on the entire test set!
# MAGIC     * **Accuracy**: During writing of this workshop, it was found that a full training session is about two minutes versus the 30 seconds that this truncated set will required.  

# COMMAND ----------

# split your training data for training and validation, but this time we have a holdout testing set already...
if False:
    # also split for training/testing 
    ratio_test = 0.2   
    # split your training data for training and validation, but this time we have a holdout testing set already...
    sdf_train_full, sdf_test = sdf_transformed.randomSplit([1 - ratio_test, ratio_test])
else:
    sdf_train_full = sdf_transformed
    sdf_test = sdf_transformed_test
    
# reduce training set from 429k to about 10k for speed during the workshop
sdf_train = sdf_train_full.sample(IHX_TRAINING_SAMPLE_FRACTION)


# COMMAND ----------


# Fit/train the model
cf, grid = create_untrained_classifier("RF", col_features)
cfModel = cf.fit(sdf_train)

# evaluate the model and show just a few columns about probabilistic and final decision values
sdf_predict = cfModel.transform(sdf_test).drop(IHX_COL_NORMALIZED)
display(sdf_predict)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Performance Evaluation
# MAGIC An important part of training a classisier is knowing when things "worked"?  In spark, these objects known as evaluators and in general data science / ML these are often called 'metrics'.  Luckily, spark has several metrics to choose from, with collections in [single binary classifiers](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.evaluation.BinaryClassificationEvaluator.html) and [ranking evaluators](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.evaluation.RankingEvaluator.html#pyspark.ml.evaluation.RankingEvaluator).
# MAGIC 
# MAGIC We'll be using a custom score metric based on ranking called ["Cumulative Gain"](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) and limit its depth to the first two deciles (or two tenths) of test data.  If you're intersting in reading more about how to create the metric, feel free to open and review the file `CUMULATIVE_GAIN_SCORE` in the utilities directory.

# COMMAND ----------

# MAGIC %run ../utilities/EVALUATOR_TOOLS

# COMMAND ----------

from pyspark.ml.evaluation import RankingEvaluator, BinaryClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.functions import vector_to_array

evaluator = evaluator_obj_get('CG2D')   # a workshop function from "EVALUATOR_TOOLS"
score_eval = evaluator.evaluate(sdf_predict)
num_train = sdf_train.count()

# COMMAND ----------

str_title = f"Original DCG (2-decile, {num_train} samples): {score_eval}"
fn_log(str_title)

evaluator_performance_curve(sdf_predict, str_title)    # a workshop function from "EVALUATOR_TOOLS"


# COMMAND ----------

# MAGIC %md
# MAGIC ### Challenge 4
# MAGIC Before we leave the classifier training workbook, let see how things do with another model type.  Specifically, a demo function was created that contained model generator code for a few classifiers.
# MAGIC 
# MAGIC Using the hints in this notebook, quickly build a method to accomplish these steps.
# MAGIC 1. Create the `LogisticRegression` untrained classifier
# MAGIC 2. Visualize the performance with the provided function `evaluator_performance_curve`

# COMMAND ----------

#### CHALLENGE  ####

# get the gradient boost classifier
model_test = "??"
# cf, grid = create_untrained_classifier(model_test, col_features)

# fit/train the calssifier with the training validator
# cfModel = cf.fit(???)

# now perform prediction on our test set and try again
# sdf_predict = cfModel.transform(sdf_test)
# score_eval = evaluator.evaluate(sdf_predict)

# step four - visualize performance
# str_title = f"{model_test} DCG (2-decile, {num_train} samples): {score_eval})"
# fn_log(str_title)
# evaluator_performance_curve(??)

#### CHALLENGE  ####


# COMMAND ----------

#### SOLUTION  ####

# get the gradient boost classifier
model_test = "LR"
cf, grid = create_untrained_classifier(model_test, col_features)

# fit/train the calssifier with the training validator
cfModel = cf.fit(sdf_train)

# now perform prediction on our test set and try again
sdf_predict = cfModel.transform(sdf_test)
score_eval = evaluator.evaluate(sdf_predict)

# step four - visualize performance
str_title = f"{model_test} DCG (2-decile, {num_train} samples): {score_eval})"
fn_log(str_title)
evaluator_performance_curve(sdf_predict, str_title)

#### SOLUTION  ####


# COMMAND ----------

# MAGIC %md
# MAGIC ### Extra Credit: Stratified Sampling!
# MAGIC In some cass, you will have a huge class imbalance that is sinking your classifier.  For example in a true (1) versus false (0) label scenario, you may find that the number of samples with a false label greatly out numbers those with a true label.  [Stratified sampling](https://spark.apache.org/docs/3.1.2/mllib-statistics.html#stratified-sampling) is one technique that can help improve the performance of your method by manipulating the existing training data.  To explore this problem andsome examples more carefully, visit the script `extra_credit/1d_SAMPLE_IMBALANCE`.
# MAGIC 
# MAGIC Try it out and report back -- does it improve performance for this problem?

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extra Credit: All-the-Samples!
# MAGIC Remember how we truncated training samples for speed?  In real ML development cases, you'ld never want to do that.  So for a little extra credit, try to modify the code to remove that random subsample operation and evaluate on the entire dataset.  You will be pleasantly surprised with how impactful adding more data for the learning processing can be.
# MAGIC 
# MAGIC If you do modify to test this code, these execution questions may help to keep your experiments in line with a narrative/story you could deliver to a customer. 
# MAGIC 
# MAGIC * When evaluating the addition of new data, is there ever a "peak" where more data doesn't help?
# MAGIC * Upon saving, can you modify the place where your final pipeline is trained to keep different copies of your model?
# MAGIC * Imagining you're delivering this pipeline to a customer, how impactful is the training time or volume on your model?  Would you need to present tradeoffs?

# COMMAND ----------

# MAGIC %md
# MAGIC Thanks for walking through this intro to classifier training.  We visualized a few features and built a pipeline that will learn on our transformed data into a speiicifc trained classifiers.
# MAGIC 
# MAGIC When you're ready, take a break! Then, head on to the next script `2a_CROSS_VALIDATION` that includes just a few examples of how to write your data in Spark.

# COMMAND ----------

# MAGIC %md
# MAGIC # Done with Classifier Intro!
# MAGIC Still want more or have questions about more advanced topics?  Scroll back up to this script section labeled `extra credit` to improve your pipeline and add imputing. Alternatively, there's another script in the `extra_credit` folder about data imbalance lessons that you may want to practice with.
