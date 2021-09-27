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
# MAGIC # Loading Serialized Pipelines
# MAGIC In our last notebook we created a few pipelines that will convert from raw input dataframes into vecctorized feature rows.  Let's make sure these transformers are portable by loading and testing a quick fit operation.

# COMMAND ----------

# load delta dataframe (it should be the same!)
# flip over to notebook '1B' to see how this was written if you're curious!
sdf_ihx_gold = spark.read.format('delta').load(IHX_GOLD)
sdf_ihx_gold_testing = spark.read.format('delta').load(IHX_GOLD_TESTING)
display(sdf_ihx_gold)

# COMMAND ----------

# MAGIC %run ../utilities/TRANSFORMER_TOOLS

# COMMAND ----------


# load untrained model first
if True:    # for now, just use the raw vectors
    pipe_transform_untrained = transformer_load(SCRATCH_IHX_VECTORIZER_PATH, IHX_VECTORIZER_PATH)
    col_features = IHX_COL_VECTORIZED
else:    # however, you can experiment with features...
    pipe_transform_untrained = transformer_load(SCRATCH_IHX_MINMAX_PATH, IHX_MINMAX_PATH)
    # pipe_transform_untrained = transformer_load(SCRATCH_IHX_L2_PATH, IHX_NORM_L2_PATH)
    col_features = IHX_COL_NORMALIZED

if pipe_transform_untrained is None:
    fn_log("Looks like an unexpected/critical error, please make sure you have acccess to the MLWorkshop data.")


# COMMAND ----------

# attempt to train model on a tenth of the total data
sdf_filled = transformer_feature_fillna(sdf_ihx_gold)
pipe_transform = pipe_transform_untrained.fit(sdf_filled)
# print the new pipeline
fn_log(pipe_transform)
fn_log(pipe_transform.stages)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Transformed Features
# MAGIC The above cell demonstrated a number of stages for string and numerical processing. 
# MAGIC 
# MAGIC However, recall that we must `fillna()` certain columns with the right type of data (e.g. *string* and *numeric*). The cell below demonstrates the transform of input features into a dense vector.  

# COMMAND ----------

# now transform data...
display(sdf_filled)
sdf_transformed = pipe_transform.transform(sdf_filled)
sdf_transformed = sdf_transformed.select(F.col(IHX_COL_LABEL), F.col(IHX_COL_INDEX), F.col(col_features))
display(sdf_transformed.limit(10))

# do the same for our TESTING data
sdf_transformed_test = pipe_transform.transform(transformer_feature_fillna(sdf_ihx_gold_testing))

# write out if admin
if is_workshop_admin():
    sdf_transformed.write.format('delta').save(IHX_GOLD_TRANSFORMED)
    sdf_transformed_test.write.format('delta').save(IHX_GOLD_TRANSFORMED_TEST)


# COMMAND ----------

# MAGIC %md
# MAGIC # Classifier Construction
# MAGIC At this point, a pipeline to transform features has been constructed and evaluated on our raw data.  Now, the task is to train and evaluate a classifier.  One of the bigger shfits to using a framework like Databricks is that it is spark driven and can show advantages (and limitations) with its distribtued environment.
# MAGIC 
# MAGIC Although there are others, we'll first explort the set of [classifiers](https://spark.apache.org/docs/latest/ml-classification-regression.html) natively available in spark.

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, GBTClassifier
from pyspark.ml.tuning import ParamGridBuilder

# simple function to get a raw clasifier
def create_untrained_classifier(classifier="RF", col_features=col_features):
    """Return a classifier and some parameters to search over for training"""
    # https://spark.apache.org/docs/latest/ml-classification-regression.html
    if classifier == "RF":
        cf = RandomForestClassifier(featuresCol=col_features, labelCol=IHX_COL_LABEL,
                            predictionCol=IHX_COL_PREDICT_BASE.format(base="int"),
                            probabilityCol=IHX_COL_PREDICT_BASE.format(base="prob"),
                            rawPredictionCol=IHX_COL_PREDICT_BASE.format(base="raw"))
        cf.setNumTrees(100)
        cf.setMaxDepth(10)
        grid = (ParamGridBuilder()   # this "grid" specifies the same settings but for a different funcction
            .addGrid(cf.numTrees, cf.getNumTrees())
            .addGrid(cf.maxDepth, cf.getMaxDepth())
            .build()
        )
    else:
        cf = GBTClassifier(featuresCol=col_features, labelCol=IHX_COL_LABEL,
                            predictionCol=IHX_COL_PREDICT_BASE.format(base="int"),
                            probabilityCol=IHX_COL_PREDICT_BASE.format(base="prob"),
                            rawPredictionCol=IHX_COL_PREDICT_BASE.format(base="raw"))
        cf.setMaxIter(15)
        cf.setMaxDepth(2)
        grid = (ParamGridBuilder()   # this "grid" specifies the same settings but for a different funcction
            .addGrid(cf.maxIter, cf.getMaxIter())
            .addGrid(cf.maxDepth, cf.getMaxDepth()
            .build()
        )

    return cf, grid


# COMMAND ----------

# MAGIC %md
# MAGIC ### Training and Evaluation
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

# Fit/train the model
cf, grid = create_untrained_classifier("RF")
cfModel = cf.fit(sdf_train)

# evaluate the model and show just a few columns about probabilistic and final decision values
sdf_predict = cfModel.transform(sdf_test).drop(IHX_COL_NORMALIZED)
display(sdf_predict)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Performance Evaluation
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
str_title = f"Original CGD (2-decile, {num_train} samples): {score_eval}"
fn_log(str_title)

evaluator_performance_curve(sdf_predict, str_title)    # a workshop function from "EVALUATOR_TOOLS"


# COMMAND ----------

# MAGIC %md
# MAGIC ### Avoiding Overfitting
# MAGIC The classifier's doing okay but not great, with cumulative gain scores around 0.36-0.37 (the [Pinnacle leaderboard](https://pinnacle.SERVICE_SITE/usecase/id/f29e33307cb04960a6697d713b0053a8) had a baseline of 0.3816 and high performers around 0.43) respectively.
# MAGIC 
# MAGIC In our current configuration (that of sub-sampling) we may need to expose the classifier to more (or slightly more equal dsitributions of) examples from the training set.  
# MAGIC 
# MAGIC One challenge that we need to cognizant of is [overfitting](https://en.wikipedia.org/wiki/Overfitting), where the classifier tries to get all samples correct but in doing so (particularly in more statistiacl learners like Bayesian classifiers and linear egressors), it makes for a worse classifier on new data.  
# MAGIC 
# MAGIC To compensate, statified sampling (with pyspark's [sampleBy](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.sampleBy.html?highlight=stratified) function) will allow each class to have close-to-equal representation.  Genrally, clasifiers can handle large disparities, but for those that fail, a rate more than a 8-to-1 should be avoided. 
# MAGIC 
# MAGIC At the same time, we'll also use the spark "all-in-one" class, [TrainValidationSplit](https://spark.apache.org/docs/latest/ml-tuning.html#train-validation-split) for training (on a fraction of the training data) and validating on the rest (the hold-out data).  

# COMMAND ----------

# count the number of labels and convert to easier pandas dataframe
df_class_counts = (sdf_train
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
sdf_transformed_stratified = sampling_stratified(sdf_train_full, IHX_COL_LABEL, IHX_COL_INDEX, 
                                              dict_fractions, sample_truncate)
# display(sdf_transformed_sampled)

# and finally specify some hold-out for valiation
ratio_validation = 0.1
tvs = TrainValidationSplit(estimator=cf, trainRatio=(1 - ratio_validation),
                           estimatorParamMaps=grid, evaluator=evaluator,
                           parallelism=1, seed=42)
cfModel = tvs.fit(sdf_transformed_stratified)
num_train = sdf_transformed_stratified.count()

# now perform prediction on our test set and try again
sdf_predict = cfModel.transform(sdf_test)
score_eval = evaluator.evaluate(sdf_predict)

str_title = f"Stratified CGD (2-decile, {num_train} samples): {score_eval} (validation CGD: {cfModel.validationMetrics})"
fn_log(str_title)
evaluator_performance_curve(sdf_predict, str_title)

# COMMAND ----------

# MAGIC %md
# MAGIC (how to justify the above?!)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Challenge 4
# MAGIC Before we leave the classifier training workbook, let see how things do with another model type.  Specifically, a demo function was created that contained model generator code for a few classifiers.
# MAGIC 
# MAGIC Using the hints in this notebook, quickly build a method to accomplish these steps.
# MAGIC 1. Create the `GBTClassifier` untrained classifier
# MAGIC 2. Find and reuse the stratified dataframe for training
# MAGIC 3. Fit/train the classifier with the training validator (using at least 10 % of training for validation)
# MAGIC 4. Visualize the performance with the provided function `evaluator_performance_curve`

# COMMAND ----------

#### SOLUTION  ####

# step one - get the gradient boost classifier
cf, grid = create_untrained_classifier("GBT")

# step two - reuse the stratified data frame...
# sdf_transformed_sampled -- check!

# step three - fit/train the calssifier with the training validator
tvs.setEstimator(cf)
tvs.setEstimatorParamMaps(grid)
cfModel = tvs.fit(sdf_transformed_stratified)
num_train = sdf_transformed_sampled.count()

# now perform prediction on our test set and try again
sdf_predict = cfModel.transform(sdf_test)
score_eval = evaluator.evaluate(sdf_predict)

# step four - visualize performance
str_title = f"Stratified, LR CGD (2-decile, {num_train} samples): {score_eval} (validation CGD: {cfModel.validationMetrics})"
fn_log(str_title)
evaluator_performance_curve(sdf_predict, str_title)
#### SOLUTION  ####


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
# MAGIC Still want more or have questions about more advanced topics?  Scroll back up to this script section labeled `extra credit` to improve your pipeline and add imputing. 
