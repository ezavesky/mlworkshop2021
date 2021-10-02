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
# MAGIC 3. Navigate to this script (Repos -> ez2685@DOMAIN -> mlworkshop2021 -> 01_databricks_notebooks -> (solutions) -> `1c_COMPUTING_FEATURES`)
# MAGIC 4. Clone the file (either `File` -> `Clone` or above with right-click)
# MAGIC    - Alternatively, add a repo (left side menu) for the [MLWorkshop2021](https://dsair@dev.azure.com/dsair/dsair_collaborator/_git/mlworkshop2021) that is stored in [Azure DevOps](https://dev.azure.com/dsair/dsair_collaborator/_git/mlworkshop2021) and jump into your own url..
# MAGIC 5. *Pat on back* or *high five*

# COMMAND ----------

# MAGIC %run ../utilities/WORKSHOP_CONSTANTS

# COMMAND ----------

# MAGIC %md
# MAGIC # Basic ETL and Exploration
# MAGIC Feature exploration by statistics...

# COMMAND ----------

# load delta dataframe (it should be the same!)
# flip over to notebook '1B' to see how this was written if you're curious!
sdf_ihx_gold = spark.read.format('delta').load(f"{IHX_GOLD}")

sdf_ihx_gold.printSchema()
display(sdf_ihx_gold)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploring Feature Schemas
# MAGIC 
# MAGIC When we run the cell below, we can see that there are a few features that were incorrectly determined as string.  There's nothing eggregious, so we'll leave these values alone for now.
# MAGIC - lots of fields with "\_ind" in their name (indiciators that are "Yes", "No", or "Unknown")
# MAGIC   - However, because of this third state ("Unknown"), let's leave this as a string value.
# MAGIC - a few fields with "\_flag" in their name (just "Yes" or "No" binary)

# COMMAND ----------

from pyspark.sql import types as T
def get_feature_column_types(sdf_data, exclude_cols=[]):
    """
    Method to retrieve the column names requested for a feature set as well as the
    """
    feature_set_config = sdf_data.columns
    schema_types = list(sdf_data.schema)
    
    # select columns that contain numeric data
    numeric_types = [T.IntegerType, T.BooleanType, T.DecimalType, T.DoubleType, T.FloatType]
    string_types = [T.StringType]
    cols_num = [c.name for c in schema_types if (type(c.dataType) in numeric_types) and not c.name in exclude_cols ]
    cols_str = [c.name for c in schema_types if (type(c.dataType) in string_types) and not c.name in exclude_cols ]
    feature_sets = {"numeric": cols_num, "string": cols_str,
                    "invalid": list(set(feature_set_config) - set(cols_str + cols_num))}
    # just for debugging, isn't really needed; find types of those that were requested but were invalid
    feature_sets["invalid_types"] = [c.dataType for c in schema_types if c.name in feature_sets['invalid']]

    fn_log(f"Feature count: {len(feature_sets['string'] + feature_sets['numeric'])} of {len(feature_set_config)} requested features")

    return feature_sets

skip_columns = ['assignment_start_dt', IHX_COL_INDEX, IHX_COL_LABEL, 'is_train']
dict_features = get_feature_column_types(sdf_ihx_gold, skip_columns)
fn_log(f"### These columns were skipped: {dict_features['invalid']}\n\n")
fn_log(f"### Are these features strings? {dict_features['string']}\n\n")
fn_log(f"### Are these features numeric? {dict_features['numeric']}\n\n")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Simple Co-variate Plotting
# MAGIC We can do this for any feature, but let's plot a pair of distributions for string-based values versus our prediction label `final_response`.  Here, we choose `region` and `assignment_start_month` as examples.

# COMMAND ----------

import matplotlib.pyplot as plt
import ipywidgets as widgets

# we're going to plot two graphs -- one over region, one over assignment month
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
df_regions = (sdf_ihx_gold
    .groupBy('final_response', 'region').count().toPandas()
    .pivot(index='region',columns=['final_response'], values='count').fillna(0)
)
df_regions.plot.bar(ax=ax1)
ax1.set_yscale('log')   # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_yscale.html#matplotlib.axes.Axes.set_yscale
ax1.grid()

df_regions = (sdf_ihx_gold
    .groupBy('final_response', 'assignment_start_month').count().toPandas()
    .pivot(index='assignment_start_month',columns=['final_response'], values='count').fillna(0)
)
df_regions.plot.bar(ax=ax2)
ax2.set_yscale('log')   # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_yscale.html#matplotlib.axes.Axes.set_yscale
ax2.grid()


# COMMAND ----------

# MAGIC %md
# MAGIC # Building a Feature Encoder
# MAGIC In this section, we're going to encode string and numeric columns via built-in spark feature encoders.  In essence, we're converting hte individual columns into a two column: (a) one that has numerical values, (b) one that has the names of those original column sources.

# COMMAND ----------

stages_spark = []
feature_cols_list = []

# COMMAND ----------

# MAGIC %md
# MAGIC ## String Features
# MAGIC String features (or non-numeric features) generally fall into two categories: 
# MAGIC * categorical features - a series of cateogires of a single features, no intrinsic ordering (blue, purple, red)
# MAGIC * ordinal features - a series of of categories of a single feature, with relative ordering (ancient, medieval, industrial, information)
# MAGIC 
# MAGIC These features can be encoded as "one hot" (e.g. make multiple unique column for each unique value) or as a "string index".  We'll opt for the string index option (by [StringIndexer](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.StringIndexer.html)) in this section.
# MAGIC * Column=`dtv_flag`...
# MAGIC   * value "No" -> 0
# MAGIC   * value "Unkown" -> 1
# MAGIC   * value "Yes" -> 2

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

for col_name in dict_features['string']:
    col_name_int = f"{col_name}_INT"
    string_indexer = StringIndexer(inputCol=col_name, outputCol=col_name_int, stringOrderType='alphabetAsc', handleInvalid='skip')
    stages_spark.append(string_indexer)
    feature_cols_list.append(col_name_int)
fn_log(f"Count of StringIndexer candidate columns {len(feature_cols_list)} ...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Numeric Features
# MAGIC Numeric features are easy, let's just add them to a single vector (e.g. a [VectorAssembler](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.VectorAssembler.html?highlight=vectorassembler#pyspark.ml.feature.VectorAssembler)) from the columns above.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Challenge 3
# MAGIC Why do we need to crunch features into a single vector?  Underlying learning models expect continuous memory blocks and don't generally care about the input feature names (or structure).  So, at training/evaluation time, your data will be stripped and transformed into blocks of memory that can be sent to one or more worker nodes for class evaluation.
# MAGIC 
# MAGIC * Raw DataFrame Format: 
# MAGIC   * Row 1: [ Feature 1 ][ Feature 2 ][ Feature 3 ][ Feature 4 ][ Feature 5 ][ Feature 6 ] ... [ Feature N ]
# MAGIC   * Row 2: [ Feature 1 ][ Feature 2 ][ Feature 3 ][ Feature 4 ][ Feature 5 ][ Feature 6 ] ... [ Feature N ]
# MAGIC   * ...
# MAGIC * Concatenated Array Format: 
# MAGIC   * [[ Feature 1, Feature 2, Feature 3, Feature 4, Feature 5, Feature 6, ..., Feature N ]
# MAGIC   *  [ Feature 1, Feature 2, Feature 3, Feature 4, Feature 5, Feature 6, ..., Feature N ]
# MAGIC   * ...
# MAGIC   * ]
# MAGIC 
# MAGIC Using the hint above, populate the right arguments for the vector assembler...
# MAGIC 1. Use the list of input features that we've been creating...
# MAGIC 2. Inform the assmbler to **skip** bad values (we'll take care of this elsewhere)
# MAGIC 3. Output to a single column named `vectorized`

# COMMAND ----------

## CHALLENGE

from pyspark.ml.feature import VectorAssembler

# define our hole set of columns as combination of stringindexed + numeric
feature_cols_list += dict_features['numeric']
col_vectorized = "vectorized"
# note that we 'skip' invalid data, so we need to be sure and zero-fill values
# assembler = VectorAssembler(inputCols=?, outputCol=?, handleInvalid=?)
# stages_spark.append(assembler)
fn_log(f"Count of VectorAssembler candidate columns {len(feature_cols_list)} ...")

## CHALLENGE

# COMMAND ----------

## SOLUTION

from pyspark.ml.feature import VectorAssembler

# define our hole set of columns as combination of stringindexed + numeric
feature_cols_list += dict_features['numeric']
col_vectorized = "vectorized"
# note that we 'skip' invalid data, so we need to be sure and zero-fill values
assembler = VectorAssembler(inputCols=feature_cols_list, outputCol=col_vectorized, handleInvalid='skip')
stages_spark.append(assembler)
fn_log(f"Count of VectorAssembler candidate columns {len(feature_cols_list)} ...")

## SOLUTION

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extra Credit: Imputing
# MAGIC There's an important message above with the 'skip' comment: invalid values are skipped.  Of course this can be avoided if we use a function like [`fillNa()`](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.fillna.html?highlight=fillna#pyspark.sql.DataFrame.fillna) but that fill value may not make sense for all columns.  Can we do better? You betcha'.
# MAGIC 
# MAGIC Enter a method like the [Imputer](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.Imputer.html?highlight=inpute).  This class will determien the mean (or other arithmatic value) for an input column and use that to fill any missing values.  There are other extensions that also look for this invalid value ahead of time, but this is a great start for making the distribution of input features more natural.
# MAGIC 
# MAGIC **Extra Credit**: Your extra credit is to modify the steps above by adding an imputer before vectorization.  If you do this, note that you'll also remove an use of the function like `fillNa`, which would pre-empty the use of the imputer otherwise.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Storing the Feature Pipeline
# MAGIC After completing our feature processing pipeline, we'll write it!  Lucky for us, these objects are all spark-natives, so we can even persist them to cloud storage without batting an eye.  You can even check-out the individual stages (saved with a little metadata) in the [Azure Storage Continer](https://PORTAL/#blade/Microsoft_Azure_Storage/ContainerMenuBlade/overview/storageAccountId/%2Fsubscriptions%2F81b4ec93-f52f-4194-9ad9-57e636bcd0b6%2FresourceGroups%2Fblackbird-prod-storage-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Fblackbirdproddatastore/path/mlworkshop2021/etag/%220x8D9766DE75EA338%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride//defaultId//publicAccessVal/None).
# MAGIC 
# MAGIC One step you may consider is [feature normalization](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.Normalizer.html?highlight=normalizer#pyspark.ml.feature.Normalizer) to our features.  Most learning methods work best with either L1 or L2 normalization instead of [min/max scaling](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.MinMaxScaler.html?highlight=minmax#pyspark.ml.feature.MinMaxScaler).  That said, we can be good data scientists and try *all three* in our classifier evaluation stage: no normalization, L2, and min/max

# COMMAND ----------

from pyspark.ml.feature import Normalizer, MinMaxScaler
from pyspark.ml import Pipeline

pipe_vectorized = Pipeline(stages=stages_spark)

norm = Normalizer(inputCol=col_vectorized, outputCol=IHX_COL_NORMALIZED)
pipe_l2 = Pipeline(stages=stages_spark + [norm])

minmax = MinMaxScaler(inputCol=col_vectorized, outputCol=IHX_COL_NORMALIZED)
pipe_minmax = Pipeline(stages=stages_spark + [minmax])

# SPECIAL NOTE: These pipelines are **UNTRAINED**.  Training them is a simple matter of running data through the `fit()` function, but we're not doing that here for simplicity.

if is_workshop_admin():  # again, these are for the workshop admin only; hopefully you can write your own to scratch below!
    quiet_delete(IHX_VECTORIZER_PATH)
    pipe_vectorized.write().save(IHX_VECTORIZER_PATH)
    quiet_delete(IHX_MINMAX_PATH)
    pipe_minmax.write().save(IHX_MINMAX_PATH)
    quiet_delete(IHX_NORM_L2_PATH)
    pipe_l2.write().save(IHX_NORM_L2_PATH)

# attempt to write to user scratch space!
try: 
    quiet_delete(SCRATCH_IHX_VECTORIZER_PATH)
    pipe_vectorized.write().save(SCRATCH_IHX_VECTORIZER_PATH)
    quiet_delete(SCRATCH_IHX_MINMAX_PATH)
    pipe_minmax.write().save(SCRATCH_IHX_MINMAX_PATH)
    quiet_delete(SCRATCH_IHX_L2_PATH)
    pipe_l2.write().save(SCRATCH_IHX_L2_PATH)
    fn_log(f"Wrote to your scratch space, you can check it out on the portal... \n{SCRATCH_URL}")
except Exception as e:
    fn_log(f"Uh oh, did you create a user scratch space? If not, that's okay, you can use the workshop's data! (Error {e})")




# COMMAND ----------

# MAGIC %md
# MAGIC Thanks for walking through this intro to feature writing.  We visualized a few features and built a pipeline that will transform our raw data into a dense numerical feature vector for subsequent learning.
# MAGIC 
# MAGIC When you're ready, head on to the next script `1d_MODELING_AND_METRICS` that includes training a basic model in Databricks with spark.

# COMMAND ----------

# MAGIC %md
# MAGIC # Done with Features!
# MAGIC Still want more or have questions about more advanced topics?  Scroll back up to this script section labeled `extra_credit` to improve your pipeline and add imputing. 
