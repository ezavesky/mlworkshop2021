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
# MAGIC 3. Navigate to this script (Repos -> ez2685@DOMAIN -> mlworkshop2021 -> 01_databricks_notebooks -> (solutions) -> `1a_DATA_READ_AND_LIST`)
# MAGIC 4. Clone the file (either `File` -> `Clone` or above with right-click)
# MAGIC    - Alternatively, add a repo (left side menu) for the [MLWorkshop2021](https://dsair@dev.azure.com/dsair/dsair_collaborator/_git/mlworkshop2021) that is stored in [Azure DevOps](https://dev.azure.com/dsair/dsair_collaborator/_git/mlworkshop2021) and jump into your own url..
# MAGIC 5. *Pat on back* or *high five*

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Attach Notebook to Cluster
# MAGIC 
# MAGIC You're ready to start the **cloned notebook** on your **personal cluster**, right? Great!
# MAGIC 
# MAGIC 1. Look at the drop down and find the personal cluster you created and click/select.
# MAGIC     * **NOTE** Please connect to the server "CDO-MLWorkshop2021" for the workshop today!
# MAGIC 2. The combo box will update for your own cluster, click that region again and select "Start Cluster"
# MAGIC     * _Pro Tip_: The next time you start this notebook it should remember what you used so you can auto-start by trying to run a cell.
# MAGIC 3. Give the cluster a minute or two to spin up and an indicator will be a green, filled circle for a _"standard cluster"_ or a green lightening bolt for a _"high concurrency"_ cluster.
# MAGIC 4. When you're ready click on a cell and hit command-enter or click the indicator _"Run All"_ on the top ribbon to run all cells.
# MAGIC 5. *Pat on back* or *high five*

# COMMAND ----------

# MAGIC %run ../utilities/WORKSHOP_CONSTANTS

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Load
# MAGIC 
# MAGIC One advantage of the Databricks environment is that it natively speaks "cloud" with almost any data dialect (e.g. data format or protocol) you can imagine.  In the following sections, we'll read a few different formats with hints on hooking up to real ADLSv2 (Azure Data Lake Service, version 2) sources that are in-process or have already been migrated from KM (kings mountain) or RDL (research data lake)
# MAGIC 
# MAGIC - You can find a tentative list of datasources created for some [SAS Resources](https://wiki.SERVICE_SITE/x/wyjGUw) and via the [Data Platform Migration Schedul](https://INFO_SITE/:x:/r/sites/KMDataMigration-UserCommunication/_layouts/15/Doc.aspx?sourcedoc=%7B735B3B59-697D-4A7A-B6C7-1C097D34E516%7D&file=End%20User%20Communication%20-%20Data%20Migration%20Schedule.xlsx&action=default&mobileredirect=true&cid=acafda66-656a-42dd-b37d-aaef427ff9eb). 
# MAGIC - You don't need to be able to run `ls` on the cloud resources.  However, to be able to list all resources currently in the cloud stored under the CDO/Azure moniker, you should apply for the right [UPSTART Role - 'AZURE CDO DATALAKE PROD DATASET LIST '](https://AUTH_SITE/upstart-az/UPSTART/UPSTART_VIEW_ACCESS.cgi?BeEUApCoDHEQAGCEDbAMBCAjEgDSBcAvEnBFDgEGDSCCALALAVEbBsDsBXBUEpBtBHEKBvDKEtEpBgBdCHBNDSBiCTCgDRCnCdEHEQDkBtDJCuEWAGAx~AZURE+CDO+DATALAKE+PROD+DATASET+LIST~VE).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspecting the storage medium...
# MAGIC Sometimes you'll need to peek at a storage directory to explore what you're dealing with.  For example, let's see what's in your source directory to recursively remove a bad file, experiment, or subdirectory, etc.  For those purposes, you may consider using [`dbutils`](https://docs.databricks.com/_static/notebooks/dbutils.html) and the filesystem set of functions.

# COMMAND ----------

# print the contents of a directory we'll be using for this workshop
try:
    fn_log(dbutils.fs.ls(MLW_DATA_ROOT))   
except Exception as e:
    fn_log(f"If you got an error, the most likely problem permissions! '{e}'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interpreting Cloud Data URLs
# MAGIC The same resource may be accessible via different tools in different formats.  The below examples point to the same resource (the one we're using in this workshop) but allow access through different clients.
# MAGIC * ADLSg2 (databricks, data tools) - `abfss://mlworkshop2021@STORAGE/ihx/`
# MAGIC * Azure portal (browser) - `https://PORTAL/#blade/Microsoft_Azure_Storage/....`
# MAGIC * azcopy url (CLI, custom apps) - `https://blackbirdproddatastore.blob.core.windows.net/mlworkshop2021/ihx/`
# MAGIC 
# MAGIC Although you can't move or delete items, you can view, download, and upload (where permissions allow) through the Azure Portal.  Check out this [Azure Portal URL](https://PORTAL/#blade/Microsoft_Azure_Storage/ContainerMenuBlade/overview/storageAccountId/%2Fsubscriptions%2F81b4ec93-f52f-4194-9ad9-57e636bcd0b6%2FresourceGroups%2Fblackbird-prod-storage-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Fblackbirdproddatastore/path/mlworkshop2021/etag/%220x8D9766DE75EA338%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride//defaultId//publicAccessVal/None) to view the workshop's primary data store.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Python - Spark Read Examples
# MAGIC Data reads typically happen with Spark commands.
# MAGIC - CSV - needs no explination, right? consider this as your tried-and-true (but equally antiquated) data format; we advise looking to delta for dramatic speed improvements! 
# MAGIC - [ORC](https://www.learntospark.com/2020/02/how-to-read-and-write-orc-file-in-apache-spark.html) - an early hadoop-friendly format using data striping for faster operations
# MAGIC - [Delta](https://databricks.com/blog/2019/08/21/diving-into-delta-lake-unpacking-the-transaction-log.html) - a Databricks developed, open source format that optimizes written tables for fast columnar scans and includes "time machine" versioning between writes; it's also the foundation of "Delta Lake" which is a Databricks wrapper product for managing these tables
# MAGIC   - **NOTE**: We can point to other speed tests for evidence, but tables in delta format are dramatically faster to read and execute queries on versus ORC, CSV, etc.

# COMMAND ----------

# load spark data frame from CSV
sdf_ihx = spark.read.format('csv').option('header', True).option('parser', '\t').load(f"{MLW_DATA_ROOT}/ihx/IHX-training.csv")
display(sdf_ihx)

# COMMAND ----------

# load delta dataframe (it should be the same!)
# flip over to notebook 'B' to see how this was written if you're curious!
sdf_ihx_gold = spark.read.format('delta').load(f"{IHX_GOLD}")
display(sdf_ihx_gold)

# COMMAND ----------

# MAGIC %md
# MAGIC ## R - notebooks and magic commands
# MAGIC Please note that you may need to use a "magic" command (place a `%r` on on the top line of the cell) if your notebook was created as python (as this WALK notebook was).  If you create a native *R* notebook (**File** -> **New Notebook** -> **Default Language** = R) you will not need to use thesemagic commands
# MAGIC 
# MAGIC Here are some common command references for R and spark with a quick primer for some dataframe manipulation options...
# MAGIC - https://docs.databricks.com/spark/latest/sparkr/overview.html - databricks spark
# MAGIC - http://spark.apache.org/docs/latest/api/R/index.html - spark R  API
# MAGIC - https://markobigdata.com/2016/03/26/sparkr-and-r-dataframe-and-data-frame/ - quick examples
# MAGIC - beware that some functions, including 'sample', may have some other conflicts...
# MAGIC   - http://spark.apache.org/docs/latest/sparkr.html#r-function-name-conflicts
# MAGIC   
# MAGIC 
# MAGIC ### SparkR vs. SparklyR
# MAGIC There are slight differences between these two Spark APIs. If you are used to the Dplyr package your might be more at home with SparkkyR. However, Databricks will use SparkR by default. You can read about the two APIs here:
# MAGIC - https://docs.databricks.com/spark/latest/sparkr/index.html#r-apis  

# COMMAND ----------

# unfortuantely, it's not easy to read from ABFSS (Azure) sources in R, but you can make a "bridge"
# this bridge exists in the spark context, which is reset / reloaded for every restart of your notebook
sdf_ihx_gold.createOrReplaceTempView("sdf_ihx_gold")

# COMMAND ----------

# MAGIC %r 
# MAGIC # easily moving from ABFSS (cloud data sets into R can be done by temporarily registering data in a spark session)
# MAGIC require(SparkR)
# MAGIC r_jobs <- sql("SELECT jobid FROM sdf_ihx_gold limit 10")
# MAGIC display(r_jobs)
# MAGIC 
# MAGIC # Otherwise, to read a CSV file, you'd need to change the source to let databricks know
# MAGIC # sdf_R <- read.df("/mnt/ihx/IHX-training.csv"), source = "csv", header="true", inferSchema = "true")
# MAGIC # sdf_R <- read.df(result = IHX_GOLD, source="delta")
# MAGIC # display(sdf_R) 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Spark SQL - Reading
# MAGIC - Load directly into a sql table
# MAGIC - Explore metastore
# MAGIC - Run same queries as sql
# MAGIC 
# MAGIC The next few cells are sql cells. In Databricks, you can specify the cell type with `%sql`. This will tell Databricks to execute the cell using the Spark SQL abstraction. You will notice most things can either be done in a similar fashion when it comes to Spark Dataframes and Spark SQL. There are sometimes a benefit to doing one vs the other but it often boils down to preference. HQL users from Research and KM will feel very much at home when using the SQL abstraction :). 

# COMMAND ----------

# MAGIC %sql 
# MAGIC show tables like '%gold%'

# COMMAND ----------

# MAGIC %sql
# MAGIC --- Below is example to load from a spark-compatible source, but it's disabled for speed
# MAGIC --- CREATE TABLE IF NOT EXISTS `sdf_ihx_gold2` USING DELTA LOCATION "abfss://mlworkshop2021@STORAGE/ihx/IHX_gold";
# MAGIC --- Then select a few lines from this table
# MAGIC --- SELECT * FROM `sdf_ihx_gold2` LIMIT 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC --- Does it look similar to the other table?
# MAGIC SELECT * FROM `sdf_ihx_gold` LIMIT 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC --- And clean up the table that we created
# MAGIC DROP TABLE IF EXISTS `sdf_ihx_gold2`;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Temp Table Cleanup
# MAGIC These are shared clusters and notebooks themselves can be messy, so don't forget to drop your unused or completed temp tables when you're done!

# COMMAND ----------

# please don't forget to release resources if you're doing a lot in one notebook
spark.catalog.dropTempView("sdf_ihx_gold")

# COMMAND ----------

# MAGIC %md
# MAGIC # Filtering and Selecting Data
# MAGIC - During experiments, we probably don't want to view all of the data for a dataset
# MAGIC - You may want to apply filters and other operations to reduce or display your results
# MAGIC - For cleaner code and to avoid functional overlap, consider using operations like `F.col` and `F.lit`
# MAGIC   - `F.col` will make sure that your command uses a specific column from the dataframe
# MAGIC   - `F.lit` will drop in a literal string or number instead of potential confusion with a column name

# COMMAND ----------

# load our SQL functions for filteirng
from pyspark.sql import functions as F

# for cleaner code (and safety in operation, consider using F.col)
# here, we display only item sin the 'SE' (south east region)
sdf_sub = sdf_ihx_gold.filter(F.col('region') == F.lit('SE'))
display(sdf_sub)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Fitlering and selecting
# MAGIC You don't need to make a long SQL or program query to accomplish most operations.  Instead, simple select and filter operations may slide the table (aka dataframe) enough for initial insepction.
# MAGIC 
# MAGIC - the `.filter(X == Y)` will select certain rows from your table
# MAGIC - the `.select(X)` will select certain columns from your table to be used

# COMMAND ----------

# MAGIC %md
# MAGIC ### Challenge 1
# MAGIC Using the hint above, try to find some information about competitors in one region...
# MAGIC 1. Filter to those only in the South ('S') region
# MAGIC 2. Selecting only the columns 'jobid', 'region', 'assignment_start_dt', 'hsd_top_competitor_price', and 'hsd_top_competitor_name' 

# COMMAND ----------

# here's a helper on how to write a list
columns_subset = ['jobid', 'region', 'assignment_start_dt', 'hsd_top_competitor_price', 'hsd_top_competitor_name']

### CHALLENGE

# here, we display only items in the 'SE' (south east region)
sdf_sub = (sdf_ihx_gold
    .filter( )
    .select( )
)
### CHALLENGE
display(sdf_sub)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Lazy Operations
# MAGIC One of the key differences between Pandas and Spark dataframes is eager versus lazy execution. In PySpark, operations are delayed until a result is actually needed in the pipeline. [A Brief Introduction to PySpark](https://towardsdatascience.com/a-brief-introduction-to-pyspark-ff4284701873).  In practicality this means your read, filtering, and other operatinos can be compounded onto a single dataframe and they won't be executed until needed.
# MAGIC 
# MAGIC This allows spark to optimize the operations and their order internally, but it also means there may be a slight delay in execution because everything isn't loaded into memory by default.
# MAGIC 
# MAGIC For cleaner code sharing among developers, we also have these two tips.
# MAGIC   - use multiple lines to break up each operation (e.g. a select, group, filter, etc.)
# MAGIC   - if using python, wrap your whole statement in parenthesis to avoid any line-wrap issues

# COMMAND ----------

# continuing from above, let's average prices from our competitors 
sdf_prices = (sdf_sub
    .groupBy(F.col('region'))
    .agg(F.min('hsd_top_competitor_price').alias('min'), 
        F.max('hsd_top_competitor_price').alias('max'))
)

# COMMAND ----------

# the above function should have been instant because there's no access to the output yet
# the below command to display (or any others that access data) will force an execution
display(sdf_prices)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Challenge 2
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
    .groupBy( )   # group by 
    .agg(F.min('hsd_top_competitor_price').alias('min'),    # aggregation
        F.max('hsd_top_competitor_price').alias('max'), 
        F.avg( ).alias('average'))
    .filter(F.col('hsd_top_competitor_name').isNotNull())   # filter (done for you)
    .orderBy(F.col('hsd_top_competitor_name'))   # ordering by competitors (done for you)
)
## SOLUTION

display(sdf_prices)

# COMMAND ----------

# MAGIC %md
# MAGIC Thanks for walking through this intro to data reads.  We just scratched the surface for data manipulation and haven't considered other topics like data ETL, visualization, or ML.
# MAGIC 
# MAGIC When you're ready, head on to the next script `1b_DATA_WRITE_EXAMPLES` that includes just a few examples of how to write your data in Spark.

# COMMAND ----------

# MAGIC %md
# MAGIC # Done with Reads!
# MAGIC Still want more or have questions about more advanced topics?  Check out the directory `extra_credit` to find a few different notebooks that have more useful details.
# MAGIC 
# MAGIC - `1_PERSONAL_CLUSTERS` - See how to create and manager your own cluster for easier library installation (out of scope for this workshop)
# MAGIC - `1_KOALAS_AND_PANDAS` - Familiar with data science and ML in python already? Check out the spark analog to Pandas, Koalas, in this notebook.
# MAGIC - `1_DATA_IN_DEEP` - Examples of getting a token from DEEP, storing personal secrets, reading from DEEP, writing to DEEP.  This extra ccredit script is definitely not required and is often referred to as a POC (proof of concept) environment instead of a business-sustained platform.
