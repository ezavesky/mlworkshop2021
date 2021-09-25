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

from pyspark.sql import functions as F
from pyspark.ml import Pipeline

def transformer_load(path_primary, path_secondary=None):
    try:
        list_files = dbutils.fs.ls(path_primary)
    except Exception as e:
        if path_seconday is None:
            fn_log(f"Failed to load transformer from '{path_primary}', no secondary provided, aborting...")
            return None
        fn_log(f"Primary failed, attempting to load secondary transformer '{path_primary}'...")
        return transformer_load(path_secondary)
    pipe_loaded = Pipeline.read().load(path_primary)
    return pipe_loaded

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

    
from pyspark.sql import types as T
def feature_fillna_safe(sdf_data, val_numeric=0, val_string=''):
    """
    Method to safely fill a dataframe acccording to type
    """
    feature_set_config = sdf_data.columns
    schema_types = list(sdf_data.schema)
    
    # select columns that contain numeric data
    numeric_types = [T.IntegerType, T.BooleanType, T.DecimalType, T.DoubleType, T.FloatType]
    string_types = [T.StringType]
    cols_num = [c.name for c in schema_types if (type(c.dataType) in numeric_types) ]
    cols_str = [c.name for c in schema_types if (type(c.dataType) in string_types) ]
    sdf_filled = (sdf_data
        .fillna(val_numeric, cols_num)
        .fillna(val_string, cols_str)
    )
    return sdf_filled

# attempt to train model on a tenth of the total data
sdf_filled = feature_fillna_safe(sdf_ihx_gold)
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
sdf_transformed_test = pipe_transform.transform(feature_fillna_safe(sdf_ihx_gold_testing))


# COMMAND ----------

# MAGIC %md
# MAGIC # Classifier Construction
# MAGIC At this point, a pipeline to transform features has been constructed and evaluated on our raw data.  Now, the task is to train and evaluate a classifier.  One of the bigger shfits to using a framework like Databricks is that it is spark driven and can show advantages (and limitations) with its distribtued environment.
# MAGIC 
# MAGIC Although there are others, we'll first explort the set of [classifiers](https://spark.apache.org/docs/latest/ml-classification-regression.html) natively available in spark.

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder

# simple function to get a raw clasifier
def create_untrained_classifier(use_random_forest=True, param_grid=False, col_features=col_features):
    """Return a classifier and some parameters to search over for training"""
    # https://spark.apache.org/docs/latest/ml-classification-regression.html
    if use_random_forest:
        cf = RandomForestClassifier(featuresCol=col_features, labelCol=IHX_COL_LABEL,
                            predictionCol=IHX_COL_PREDICT_BASE.format(base="int"),
                            probabilityCol=IHX_COL_PREDICT_BASE.format(base="prob"),
                            rawPredictionCol=IHX_COL_PREDICT_BASE.format(base="raw"))
        cf.setNumTrees(100)
        cf.setMaxDepth(10)
        grid = (ParamGridBuilder()   # this "grid" specifies the same settings but for a different funcction
            .addGrid(cf.numTrees, [100])
            .addGrid(cf.maxDepth, [10])
            .build()
        )
    else:
        cf = LogisticRegression(featuresCol=col_features, labelCol=IHX_COL_LABEL,
                                predictionCol=IHX_COL_PREDICT_BASE.format(base="int"),
                                probabilityCol=IHX_COL_PREDICT_BASE.format(base="prob"),
                                rawPredictionCol=IHX_COL_PREDICT_BASE.format(base="raw"))
        # We can also use the multinomial family for binary classification
        cf.setMaxIter(15)
        cf.setTol(1E-6)    
        cf.setRegParam(0.1)
        cf.setElasticNetParam(0.8)
        cf.setFamily("multinomial")
        grid = (ParamGridBuilder()   # this "grid" specifies the same settings but for a different funcction
            .addGrid(cf.maxIter, [15])
            .addGrid(cf.tol, [1e-6])
            .addGrid(cf.regParam, [0.1])
            .addGrid(cf.elasticNetParam, [0.8])
            .build()
        )

    return cf, grid


# COMMAND ----------

# MAGIC %md
# MAGIC ### Training and Evaluation
# MAGIC In these classifiers, the basic workflow is parameter setting, training (executed with `fit()`) and evaluation (executed with `transform()`).  Let's see how it works with two common classifiers, [Random Forests](https://spark.apache.org/docs/latest/ml-classification-regression.html#random-forest-classifier) and [Logistic Regression](https://spark.apache.org/docs/latest/ml-classification-regression.html#multinomial-logistic-regression).  
# MAGIC 
# MAGIC One other observation is that normally you'd have to split your training data into a training set and a testing (or holdout) set.  To keep things similar to real-world data, you should make the sample choice random.

# COMMAND ----------

# normally, split your training data, but this time we have a holdout testing set already...
if False:
    # also split for training/testing 
    ratio_test = 0.2
    sdf_train, sdf_test = sdf_transformed.randomSplit([1 - ratio_test, ratio_test])
else:
    sdf_train = sdf_transformed
    sdf_test = sdf_transformed_test

# Fit/train the model
cf, grid = create_untrained_classifier(True)
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

# MAGIC %run ../utilities/CUMULATIVE_GAIN_SCORE

# COMMAND ----------

from pyspark.ml.evaluation import RankingEvaluator, BinaryClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.functions import vector_to_array

# get the distributed evaluator type from spark for evaluation
def get_evaluator_obj(type_match='CGD'):
    if type_match == "mAP":
        evalObj = RankingEvaluator(predictionCol=IHX_COL_PREDICT_BASE.format(base="score"),
                                   labelCol=IHX_COL_PREDICT_BASE.format(base="label"), 
                                   metricName='meanAveragePrecision', k=100)
    elif type_match == "AUC":
        # wrap_value = F.udf(lambda v: [ [ v ] ], T.ArrayType(T.FloatType()))
        evalObj = BinaryClassificationEvaluator(rawPredictionCol=IHX_COL_PREDICT_BASE.format(base="int"),
                                                labelCol=IHX_COL_LABEL, metricName='areaUnderROC')
    else:
        evalObj = CuluativeGainEvaluator(rawPredictionCol=IHX_COL_PREDICT_BASE.format(base="prob"), 
                                         labelCol=IHX_COL_LABEL, metricName='CG2D')
    return evalObj


evaluator = get_evaluator_obj('CGD')
score_eval = evaluator.evaluate(sdf_predict)
str_title = f"CGD (2nd decile): {score_eval}"
fn_log(str_title)


# # let's also get a confusion matrix...
def get_performance_curve(sdf_predict, str_title=None):
    from sklearn import __version__ as sk_versions
    ver_parts = sk_versions.split('.')
    is_new_sk = int(ver_parts[0]) >= 1 or int(ver_parts[1]) >= 24
    df_predict = (sdf_predict
        .withColumn('score', udf_last_prediction(F.col(IHX_COL_PREDICT_BASE.format(base="prob"))))
        .select('score', F.col(IHX_COL_LABEL).alias('label'))
        .toPandas()
    )
    # if we were looking at a cclassificcation problem....
    # from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    # df_predict = sdf_predict.select(IHX_COL_PREDICT_BASE.format(base="int"), IHX_COL_LABEL).toPandas()
    # cm = confusion_matrix(df_predict[IHX_COL_LABEL], df_predict[IHX_COL_PREDICT_BASE.format(base="int")])
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["declined", "engaged"])

    if not is_new_sk:   # scikit v0.24 yet?
        from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
        precision, recall, _ = precision_recall_curve(df_predict['label'], df_predict['score'])
        disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    else:
        from sklearn.metrics import DetCurveDisplay
        disp = DetCurveDisplay.from_predictions(df_predict['score'], df_predict['label']) 
    #                                             display_labels=["declined", "engaged"])
    disp.plot()
    disp.ax_.grid()
    if str_title is not None:  # add title if we need to
        disp.ax_.set_title(str_title)


get_performance_curve(sdf_predict, str_title)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Avoiding Overfitting
# MAGIC The classifier's not doing well at all, scoring an average precision of 0.5 (usable values are 0.8 or above) and declaring that everything is a 'decline' class, what gives? One challenge that need to cognizant of is [overfitting](https://en.wikipedia.org/wiki/Overfitting), where the classifier tries to get all samples correct but in doing so, it makes for a worse classifier on new data.  
# MAGIC 
# MAGIC To compensate, statified sampling (with pyspark's [sampleBy](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.sampleBy.html?highlight=stratified) function) will allow each class to have close-to-equal representation.  Genrally, clasifiers can handle some disparities, but you neeed to exerccise caution for more than a 5-to-1 ratio should be avoided. 
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

# now, let's compute a sampling ratio (no greater than 4-to-1) for stratified sampling
max_imbalance = 10.0
min_count = df_class_counts['count'].min() * max_imbalance
dict_fractions = df_class_counts.set_index(IHX_COL_LABEL).to_dict(orient='index')
fn_log(dict_fractions)
dict_fractions = {class_key: (min(min_count, dict_fractions[class_key]['count'])/dict_fractions[class_key]['count']) 
                  for class_key in dict_fractions}
fn_log(f"Min Class Count {min_count}, Max imbalance: {max_imbalance}, Sample ratio: {dict_fractions}")

# COMMAND ----------

from pyspark.ml.tuning import TrainValidationSplit

def stratified_sample(sdf, col_name, col_unique, dict_fractions):
    """Quick custom function for stratified sampling...; dict_fractions = {1:0.4, 0:0.6}"""
    # this is the line that we want to execute, but as of 9/23 some security settings prevent it!
    # sdf_transformed_sampled = sdf.sampleBy(F.col(col_unique), fractions=dict_fractions, seed=0)
    sdf_working = sdf.withColumn('_salt', F.abs(F.hash(col_unique)) % 100)
    # loop through keys to compile our own dataframe
    for key_idx in dict_fractions:
        # apply a filter that will select/match the key and limit the number of samples
        salt_filter = int(dict_fractions[key_idx]*100)
        fn_log(f"[stratified_sample] Filter '{col_name}'=={key_idx}, limit: {salt_filter} %")
        sdf_working = sdf_working.filter((F.col(col_name) != key_idx) | (F.col('_salt') <= salt_filter))
    return sdf_working.drop(F.col('_salt'))      # all done? drop the random column we generated

# apply our stratified sampled
sdf_transformed_sampled = stratified_sample(sdf_train, IHX_COL_LABEL, IHX_COL_INDEX, dict_fractions)
# display(sdf_transformed_sampled)

# and finally specify some hold-out for valiation
ratio_validation = 0.25
tvs = TrainValidationSplit(estimator=cf, trainRatio=(1 - ratio_validation),
                           estimatorParamMaps=grid, evaluator=evaluator,
                           parallelism=1, seed=42)
cfModel = tvs.fit(sdf_transformed_sampled)

# now perform prediction on our test set and try again
sdf_predict = cfModel.transform(sdf_test)

str_title = f"CGD (2nd decile): {cfModel.validationMetrics}"
fn_log(str_title)
get_performance_curve(sdf_predict, str_title)

# COMMAND ----------

# vector_assembler_trained = pipeline_trained.stages[-1]
# vector_assembler_trained = vector_assembler_trained.stages[0]    # peek in to get input features
# feature_cols_list = vector_assembler_trained.getInputCols()   # get the column names from the assembler
# fn_log(f"[{feature_set}] Found multi-stage assebly, {pipeline_trained.stages[-1]} with assembler {vector_assembler_trained}...")

#  Fit the model
# cfModel = cf.fit(sdf_transformed)

# sdf_predict = cfModel.transform(sdf_transformed).drop(IHX_COL_NORMALIZED)
# display(sdf_predict)



# COMMAND ----------

# MAGIC %md
# MAGIC 1. Load the transfomer
# MAGIC 3. Evaluate with a simple logistic regressor
# MAGIC    - Challenge: update to use the min/max
# MAGIC 2. Create a standard evaluator (example)
# MAGIC 4. Update to use a custom evaluator
# MAGIC 5. Write out predictions, model, and performance to scratch location

# COMMAND ----------

# MAGIC %md
# MAGIC ### Challenge 4
# MAGIC The `groupBy` function selects one or more rows to group with and `agg` selects a [https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.functions.aggregate.html](function) from a large [available list](https://sparkbyexamples.com/pyspark/pyspark-aggregate-functions/).
# MAGIC 
# MAGIC Using the hint above, let's quickly find aggregate prices across all the datal with a little more detail...
# MAGIC 1. For all regions (the original dataset)...
# MAGIC 2. Use the competitor (e.g. `hsd_top_competitor_name`) as a grouping column as well...
# MAGIC 3. Find the average price for each competitor?
# MAGIC 4. Filter out those columns that are null/empty.  *(this is provided for you)*
# MAGIC 5. Sort by our competitor names.  *(this is provided for you)*

# COMMAND ----------

# MAGIC %md
# MAGIC Thanks for walking through this intro to feature writing.  We visualized a few features and built a pipeline that will transform our raw data into a dense numerical feature vector for subsequent learning.
# MAGIC 
# MAGIC When you're ready, head on to the next script `1b_DATA_WRITE_EXAMPLES` that includes just a few examples of how to write your data in Spark.

# COMMAND ----------

# MAGIC %md
# MAGIC # Done with Features!
# MAGIC Still want more or have questions about more advanced topics?  Scroll back up to this script section labeled `extra_credit` to improve your pipeline and add imputing. 
