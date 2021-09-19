# Databricks notebook source
# MAGIC %md
# MAGIC # Part 1 - MLWorkshop2021
# MAGIC This noteobook is part of the [2021 ML Workshop](https://INFO_SITE/cdo/events/internal-events/90dea45e-1454-11ec-8dca-7d45a6b8dd2a) (sponsored by the Software Symposium)

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
# MAGIC 2. The combo box will update for your own cluster, click that region again and select "Start Cluster"
# MAGIC   * _Pro Tip_: The next time you start this notebook it should remember what you used so you can auto-start by trying to run a cell.
# MAGIC 3. Give the cluster a minute or two to spin up and an indicator will be a green, filled circle for a _"standard cluster"_ or a green lightening bolt for a _"high concurrency"_ cluster.
# MAGIC 4. When you're ready click on a cell and hit command-enter or click the indicator _"Run All"_ on the top ribbon to run all cells.
# MAGIC 5. *Pat on back* or *high five*

# COMMAND ----------

# MAGIC %run ./0_WORKSHOP_CONSTANTS

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
    print(dbutils.fs.ls(MLW_DATA_ROOT))   
except Exception as e:
    print(f"If you got an error, the most likely problem permissions! '{e}'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Python - Spark Read Examples
# MAGIC Data reads typically happen with Spark commands.
# MAGIC - CSV - needs no explination, right? consider this as your tried-and-true (but equally antiquated) data format; we advise looking to delta for dramatic speed improvements! 
# MAGIC - [ORC](https://www.learntospark.com/2020/02/how-to-read-and-write-orc-file-in-apache-spark.html) - an early hadoop-friendly format using data striping for faster operations
# MAGIC - [Delta](https://databricks.com/blog/2019/08/21/diving-into-delta-lake-unpacking-the-transaction-log.html) - a Ddatabricks developed, open source format that optimizes written tables for fast columnar scans and includes "time machine" versioning between writes; it's also the foundation of "Delta Lake" which is a Databricks wrapper product for managing these tables
# MAGIC   - **NOTE**: We can point to other speed tests for evidence, but tables in delta format are dramatically faster to read and execute queries on versus ORC, CSV, etc.

# COMMAND ----------

# load spark data frame from CSV
sdf_ihx = spark.read.format('csv').option('header', True).load(f"{MLW_DATA_ROOT}/ihx/IHX-training.csv")
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
sdf_ihx.createOrReplaceTempView("sdf_ihx")

# COMMAND ----------

# MAGIC %r 
# MAGIC # easily moving from ABFSS (cloud data sets into R can be done by temporarily registering data in a spark session)
# MAGIC require(SparkR)
# MAGIC r_jobs <- sql("SELECT jobid FROM sdf_ihx")
# MAGIC display(r_jobs)
# MAGIC 
# MAGIC # Otherwise, to read a CSV file, you'd need to change the source to let databricks know
# MAGIC # sdf_R <- read.df("/mnt/ihx/IHX-training.csv"), source = "csv", header="true", inferSchema = "true")
# MAGIC # sdf_R <- read.df(result = IHX_GOLD, source="delta")
# MAGIC # display(sdf_R) 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interpreting Cloud Data URLs
# MAGIC The same resource may be accessible via different tools in different formats.  The below examples point to the same resource (the one we're using in this workshop) but allow access through different clients.
# MAGIC * ADLSg2 (databricks, data tools) - `abfss://mlworkshop2021@STORAGE/`
# MAGIC * Azure portal (browser) - `https://PORTAL/#blade/Microsoft_Azure_Storage/....`
# MAGIC * azcopy url (CLI, custom apps) - `https://blackbirdproddatastore.blob.core.windows.net/mlworkshop2021`
# MAGIC 
# MAGIC Although you can't move or delete items, you can view, download, and upload (where permissions allow) through the Azure Portal.  Check out this [Azure Portal URL](https://PORTAL/#blade/Microsoft_Azure_Storage/ContainerMenuBlade/overview/storageAccountId/%2Fsubscriptions%2F81b4ec93-f52f-4194-9ad9-57e636bcd0b6%2FresourceGroups%2Fblackbird-prod-storage-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Fblackbirdproddatastore/path/mlworkshop2021/etag/%220x8D9766DE75EA338%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride//defaultId//publicAccessVal/None) to view the workshop's primary data store.

# COMMAND ----------

# MAGIC %md
# MAGIC # Filtering and Selecting Data
# MAGIC - During experiments, we probably don't want to view all of the data for a dataset
# MAGIC - You may want to apply filters and other operations to reduce or display your results

# COMMAND ----------

# load our SQL functions for filteirng
from pyspark.sql import functions as F

# load delta dataframe (it should be the same!)
sdf_ihx_gold = spark.read.format('delta').load(f"{IHX_GOLD}")
display(sdf_ihx_gold)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fitlering and selecting
# MAGIC You don't need to make a long SQL or program query to accomplish most operations.  Instead, simple select and filter operations may slide the table (aka dataframe) enough for initial insepction.
# MAGIC 
# MAGIC - the `.filter(X == Y)` will select certain rows from your table
# MAGIC - the `.select(X)` will select certain columns from your table to be used

# COMMAND ----------

# subsetting to rows where final_network_name is "Showtime"
# TODO: how do we print out just showtime?

# (uncomment + fix)
# showtime = eview_delta.filter(XX == YY)

# give up or need a hint? look at and expand the cell below!

# COMMAND ----------

#### ANSWER KEY for filter of showtime #####
showtime = eview_delta.filter(col('final_network_name') == 'Showtime')
# some examples will wrap a string or numerical value in `lit` which means "literal" (not sure why?)
showtime = eview_delta.filter(col('final_network_name') == lit('Showtime'))

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Lazy Operations

# COMMAND ----------

display(showtime)

# COMMAND ----------

grpby = showtime.groupby('program_title').count()

# COMMAND ----------

display(grpby)

# COMMAND ----------

# group by showtime programs and count number of viewing records
# play around with display features
# - count desc/asc
# - different plot options
display(grpby.orderBy('count', ascending=False).limit(15))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Koalas Examples
# MAGIC If you're in python and have worked with data in ML libraries, you may be more familar with the library [`pandas`](https://pandas.pydata.org/docs/).  While there is compatibility between spark data frames and pandas (e.g. just call the funciton [`df.toPandas()`](https://docs.databricks.com/spark/latest/spark-sql/spark-pandas.html)), there is another Spark-native way to manipulate content with the pandas API.
# MAGIC 
# MAGIC Enter [Koalas](https://docs.databricks.com/languages/koalas.html).  Yes, notn only a helpful data management package, but another adorable animal you might see in the zoo (or an [Aussie's home](https://thehill.com/homenews/news/476429-zoo-director-saves-pandas-monkeys-from-australia-fire-by-taking-them-home)).
# MAGIC - developed by databricks, so spark native as long as possible
# MAGIC - same friendly API for a lot of select, group and manipulation functions

# COMMAND ----------

import databricks.koalas as ks

# convert from a spark dataframe
kdf_showtime = ks.DataFrame(grpby)

# take a subset after sorting by count
# https://koalas.readthedocs.io/en/latest/reference/api/databricks.koalas.DataFrame.sort_values.html
kdf_sub = kdf_showtime.sort_values('count', ascending=False).head(100)
print("## TOP 10 most viewed ##")
print(kdf_sub.head(10))

# use pandas-like column accessors for a column (resort by title)
print("## First 10 alphabetical titles in top 100 ")
kdf_mini = kdf_sub.sort_values('program_title', ascending=True)
print(kdf_mini['program_title'].head(10))

# COMMAND ----------

# transorm between pandas and spark with ease
print("LIFE AS A PANDAS DATAFRAME...", kdf_mini.to_pandas())
print("LIFE AS A SPARK DATAFRAME...", kdf_mini.to_spark())

# or even other data-repesentative formats
print("LIFE AS JSON...", kdf_mini.head(3).to_json(orient='records'))



# COMMAND ----------

# load a quick delta chunk for another test...
kdf_eview = ks.DataFrame(eview_delta.limit(100))
display(kdf_eview)

# COMMAND ----------

# what about filtering (as we did with spark?)
# HINT: spark was this ... eview_delta.filter(col('final_network_name') == 'Showtime')

# subsetting to rows where final_network_name is "Showtime"
# TODO: how do we print out just showtime?
kdf_eview = ks.DataFrame(eview_delta)
## uncomment and fix!
## display(kdf_eview[kdf_eview[XX] == YY)



# COMMAND ----------

# ANSWER -- there are several ways to get this done, just like in pandas
display(kdf_eview[kdf_eview['final_network_name'] == "Showtime"].head(5))
print(kdf_eview.where(kdf_eview['final_network_name'] == "Showtime"))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Spark SQL Examples
# MAGIC - Load directly into a sql table
# MAGIC - Explore metastore
# MAGIC - Run same queries as sql
# MAGIC 
# MAGIC The next few cells are sql cells. In Databricks, you can specify the cell type with `%sql`. This will tell Databricks to execute the cell using the Spark SQL abstraction. You will notice most things can either be done in a similar fashion when it comes to Spark Dataframes and Spark SQL. There are sometimes a benefit to doing one vs the other but it often boils down to preference. HQL users from Research and KM will feel very much at home when using the SQL abstraction :). 

# COMMAND ----------

# MAGIC %sql 
# MAGIC show tables

# COMMAND ----------

# MAGIC %md
# MAGIC **Create a delta table in the Databricks metastore directly from source data:**

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS  eview_20210308 USING DELTA LOCATION 'abfss://eview-delta@dsairgeneraleastus2sa.STORAGE/program_watch_start_date=20210308';

# COMMAND ----------

# MAGIC %sql 
# MAGIC show tables

# COMMAND ----------

# MAGIC %md 
# MAGIC **We can create temporary views that store data for intermediary steps, without writing it out somewhere (NOTE: will not persist)**

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE OR REPLACE TEMPORARY VIEW tmp_showtime as select * from eview_20210308 where final_network_name = 'Showtime'

# COMMAND ----------

# MAGIC %sql
# MAGIC show views

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from tmp_showtime limit 10

# COMMAND ----------

# MAGIC %md
# MAGIC #### Cleanup
# MAGIC Tables loaded as SQL (e.g. with "metadata" and viewable via hte left side) do use up memory.  So, consider deleting and unloading those that you don't use.  What's worse, when working on the shared compute clusters, those tables may confuse others or add clutter to the workspace, which may get frustrating if not kept tidy.

# COMMAND ----------

# MAGIC %sql 
# MAGIC select program_title, count(*) as view_count from tmp_showtime group by program_title

# COMMAND ----------

# MAGIC %sql
# MAGIC select program_title, count(*) as view_count from tmp_showtime group by program_title order by view_count desc limit 15
