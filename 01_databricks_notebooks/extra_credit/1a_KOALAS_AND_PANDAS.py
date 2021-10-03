# Databricks notebook source
# MAGIC %md
# MAGIC # Koalas Examples
# MAGIC If you're in python and have worked with data in ML libraries, you may be more familar with the library [`pandas`](https://pandas.pydata.org/docs/).  While there is compatibility between spark data frames and pandas (e.g. just call the funciton [`df.toPandas()`](https://docs.databricks.com/spark/latest/spark-sql/spark-pandas.html)), there is another Spark-native way to manipulate content with the pandas API.
# MAGIC 
# MAGIC Enter [Koalas](https://docs.databricks.com/languages/koalas.html).  Yes, not only a helpful data management package, but another adorable animal you might see in the zoo (or an [Aussie's home](https://thehill.com/homenews/news/476429-zoo-director-saves-pandas-monkeys-from-australia-fire-by-taking-them-home)).
# MAGIC - developed by databricks, so spark native as long as possible
# MAGIC - same friendly API for a lot of select, group and manipulation functions

# COMMAND ----------

# MAGIC %run ../utilities/WORKSHOP_CONSTANTS

# COMMAND ----------

# load delta dataframe (it should be the same!)
# flip over to notebook 'B' to see how this was written if you're curious!
sdf_ihx_gold = spark.read.format('delta').load(f"{IHX_GOLD}")
display(sdf_ihx_gold)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Operations
# MAGIC Koalas was coded to more easily transition those familiar with Pandas, so a lot of the syntax is similar.  Here are some quick examples that use our spark dataframe -- but with the huge advantage that we don't have to load everything into memory!

# COMMAND ----------

import databricks.koalas as ks
from pyspark.sql import functions as F

# convert from a spark dataframe
kdf_ihx_hold = ks.DataFrame(sdf_ihx_gold)
print("## COLUMNS (only the first 10) ##")
print(list(kdf_ihx_hold.columns)[:10])
list_interesting_cols = ['jobid', 'region', 'assignment_start_dt', 'hsd_top_competitor_name', 'hsd_top_competitor_price']
print("")

# take a subset after sorting by count
# https://koalas.readthedocs.io/en/latest/reference/api/databricks.koalas.DataFrame.sort_values.html
kdf_sub = kdf_ihx_hold.sort_values('assignment_start_dt', ascending=False).head(1000)
print("## TOP 10 most recent jobs ##")
print(kdf_sub[list_interesting_cols].head(10))
print("")

# use pandas-like column accessors for a column (resort by title)
# note that we start from the output of the last operation, which already pulled the recent jobs
print("## Most expensive competitors from the most reccent 1000 jobs")
kdf_mini = kdf_sub.sort_values('hsd_top_competitor_price', ascending=False)
print(kdf_mini[list_interesting_cols].head(10))
print("")


# COMMAND ----------

# transorm between pandas and spark with ease
print("LIFE AS A PANDAS DATAFRAME...", kdf_mini.to_pandas())
print("LIFE AS A SPARK DATAFRAME...", kdf_mini.to_spark())

# or even other data-repesentative formats
print("LIFE AS JSON...", kdf_mini.head(3).to_json(orient='records'))



# COMMAND ----------

# MAGIC %md
# MAGIC ## Challenge 2 (revisted)
# MAGIC 
# MAGIC This challenge was in the notebook `1a_DATA_READ_AND_LIST`...
# MAGIC 
# MAGIC The `groupBy` function selects one or more rows to group with and `agg` selects a [https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.functions.aggregate.html](function) from a large [available list](https://sparkbyexamples.com/pyspark/pyspark-aggregate-functions/).
# MAGIC 
# MAGIC Using the hint above, let's quickly find aggregate prices across all the datal with a little more detail...
# MAGIC 1. For all regions (the original dataset)...
# MAGIC 2. Use the competitor (e.g. `hsd_top_competitor_name`) as a grouping column as well...
# MAGIC 3. Find the average price for each competitor?
# MAGIC 4. Filter out those columns that are null/empty.  *(this is provided for you)*
# MAGIC 5. Sort by our competitor names.  *(this is provided for you)*

# COMMAND ----------

## SOLUTION

# continuing from above, let's average prices from our competitors 
sdf_prices = (sdf_ihx_gold
    .groupBy(F.col('region'), F.col('hsd_top_competitor_name'))   # group by 
    .agg(F.min('hsd_top_competitor_price').alias('min'),    # aggregation
        F.max('hsd_top_competitor_price').alias('max'), 
        F.avg('hsd_top_competitor_price').alias('average'))
    .filter(F.col('hsd_top_competitor_name').isNotNull())   # filter
    .orderBy(F.col('hsd_top_competitor_name'))
)
## SOLUTION

display(sdf_prices)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Challenge 2 (koalas!)
# MAGIC 
# MAGIC Okay, you know what it should look like, so try to recreate the whole thing in koalas!

# COMMAND ----------

## YOUR ATTEMPT?

kdf_ihx_hold = ks.DataFrame(sdf_ihx_gold)
kdf_sub1 = (kdf_ihx_hold
)
# kdf_sub1.columns =  # apply clean column names?
kdf_sub2 = (kdf_sub1
)
display(kdf_sub1)

# COMMAND ----------

##### SOLUTION 
# (scroll down to see the answer!)






























kdf_ihx_hold = ks.DataFrame(sdf_ihx_gold)
kdf_sub1 = (
    kdf_ihx_hold.groupby(['region', 'hsd_top_competitor_name'])  # group
    .agg({'hsd_top_competitor_price':['min', 'max', 'avg']})   # agg on one column
    .reset_index()   # get original group index
)
kdf_sub1.columns = ['region', 'competitor', 'min', 'max', 'avg']   # apply clean column names
kdf_sub2 = (
    kdf_sub1.dropna(subset=['competitor'])
    .sort_values(['competitor'])
)
display(kdf_sub1)
