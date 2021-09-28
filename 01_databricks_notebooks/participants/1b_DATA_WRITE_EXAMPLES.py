# Databricks notebook source
# MAGIC %md
# MAGIC # Part 1 - MLWorkshop2021
# MAGIC This notebook is part of the [2021 ML Workshop](https://INFO_SITE/cdo/events/internal-events/90dea45e-1454-11ec-8dca-7d45a6b8dd2a) (sponsored by the Software Symposium)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clone This Demo Notebook
# MAGIC 1. Head to http://FORWARD_SITE/cdo-databricks  (don't worry, it's a fowarding DNS to the main CDO PROD(uction) instance)
# MAGIC 2. Log-in via your AT&T ID and DOMAIN password
# MAGIC 3. Navigate to this script (Repos -> ez2685@DOMAIN -> mlworkshop2021 -> 01_databricks_notebooks -> (solutions) -> `1b_DATA_WRITE_EXAMPLES`)
# MAGIC 4. Clone the file (either `File` -> `Clone` or above with right-click)
# MAGIC    - Alternatively, add a repo (left side menu) for the [MLWorkshop2021](https://dsair@dev.azure.com/dsair/dsair_collaborator/_git/mlworkshop2021) that is stored in [Azure DevOps](https://dev.azure.com/dsair/dsair_collaborator/_git/mlworkshop2021) and jump into your own url..
# MAGIC 5. *Pat on back* or *high five*

# COMMAND ----------

# MAGIC %run ../utilities/WORKSHOP_CONSTANTS

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get your own Scratch!
# MAGIC The most important lesson here is to make sure you're writing company data (even your own experiments) into a location that is...
# MAGIC - protected, 
# MAGIC - persistent, and
# MAGIC - cloud-accessible!
# MAGIC 
# MAGIC Need your own scratch space? Follow along with the presentation or head over to the [self service CDO page](https://INFO_SITE/sites/data/SitePages/Creating-an-ADLSGen2-Scratch-Space.aspx) to understand how to get your own slice!

# COMMAND ----------

# MAGIC %md
# MAGIC # Writing to cloud storage
# MAGIC Writing data is simple task, but there are some unique properties / effects that are available to those using spark.
# MAGIC 
# MAGIC 1. Data order (read and write) is not guaranteed during a write
# MAGIC 2. Data can be easily appeneded during a write
# MAGIC 3. Storage accounts may not behave like file systems, with these few examples as most notable.
# MAGIC    - File referneces are key/value pairs, not hierarchical storage
# MAGIC    - You can't make empty subdirectories in most cloud storage (especially Azure storage objects)
# MAGIC    - You create a new path by writing to it, all parents that don't exist will be "created"
# MAGIC 4. If you have one or more indexes, you can greatly speed up the write/read of your data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Partitions
# MAGIC The last point is perhaps the most important as we talk about **BIG** data. In the example below, we read from the CSV and write to a `delta` format file to show non-partitioned and a partition using the `region` as a primary index.

# COMMAND ----------

# this code is commented out because it's just provided for reference...
#   namely, how we got the IHX data into an ADLSg2 storage container
# NOTE: You may not be able to run the first command this due to privilages, but if you've created a 
IHX_GOLD_UNPARTITIONED = f"{IHX_GOLD}-unpartitioned"
if is_workshop_admin():
    # you may need to delete a prior version because delta will otherwise optimize it as an update
    try:
        dbutils.fs.rm(IHX_GOLD, True)   # recursively delete what was there before
        dbutils.fs.rm(IHX_GOLD_TESTING, True)
        dbutils.fs.rm(IHX_GOLD_UNPARTITIONED, True)
    except Excpetion as e:
        fn_log("Error clearing parititon, maybe it didn't exist...")

    # example of loading CSV and writing to a ADLSg2 location in delta format
    # NOTE the critical 'inferSchema' option, which will attempt to transform into more than string types
    # look here for more options... https://github.com/databricks/spark-csv

    # write to the space used by the workshop with no parition configuraiton
    sdf_ihx = (
        spark.read.format('csv')
        .option('nullValue', 'NULL')
        .option("inferSchema", "true")
        .option("sep", "\t")
        .option('header', True)
        .load(f"{MLW_DATA_ROOT}/ihx/IHX-training.csv")
    )
    sdf_ihx.write.format('delta').mode('overwrite').save(f"{IHX_GOLD_UNPARTITIONED}")
    # scratch space under your user ID, the seccond command should work.
    (sdf_ihx.repartition('assignment_start_month')
         .write.format('delta').partitionBy('assignment_start_month')
         .mode('overwrite').save(f"{IHX_GOLD}"))

    # now read and repartition the testing data as well
    sdf_ihx_testing = (
        spark.read.format('csv')
        .option('nullValue', 'NULL')
        .option("inferSchema", "true")
        .option("sep", "\t")
        .option('header', True)
        .load(f"{MLW_DATA_ROOT}/ihx/IHX-testing.csv")
    )    
    # scratch space under your user ID, the seccond command should work.
    (sdf_ihx_testing.repartition('assignment_start_month')
         .write.format('delta').partitionBy('assignment_start_month')
         .mode('overwrite').save(f"{IHX_GOLD_TESTING}"))


# COMMAND ----------

list_files = dbutils.fs.ls(IHX_GOLD_UNPARTITIONED)
fn_log(f"## Files in non-partitioned format... {len(list_files)} total from path '{IHX_GOLD_UNPARTITIONED}'")
for file_ref in list_files[:5]:   # just pring the first five
    fn_log(f"{file_ref.name}: {file_ref.size} bytes")
fn_log("")
    
list_files = dbutils.fs.ls(IHX_GOLD)
fn_log(f"## Files in month-partitioned format... {len(list_files)} total from path '{IHX_GOLD}'")
for file_ref in list_files[:5]:   # just pring the first five
    fn_log(f"{file_ref.name}: {file_ref.size} bytes")

# COMMAND ----------

path_sub = f"{IHX_GOLD}/{list_files[-1].name}"
list_files_sub = dbutils.fs.ls(path_sub)
fn_log(f"## Files in region-partitioned format... {len(list_files_sub)} total from path '{path_sub}'")
for file_ref in list_files_sub[:5]:   # just pring the first five
    fn_log(f"{file_ref.name}: {file_ref.size} bytes")

# COMMAND ----------

# MAGIC %md
# MAGIC So what we observe is that the partition creates subfolders and forces the data for a certain value to be grouped into one partition.  It looks like the file sizes are different and ther are fewer partitions, but if, as SMEs of the data, we have intuition about a natural data partition it may be advantagous to use it.  
# MAGIC 
# MAGIC For example, in an ever-growing logging, telemtry, or behavioral dataset, it may make the most sense to use one of the time-based fields as a primary index.  We plotted that distribution using the `assignment_start_month` field below.

# COMMAND ----------

from pyspark.sql import functions as F
sdf_ihx = spark.read.format('delta').load(f"{IHX_GOLD}")
display(sdf_ihx
   .groupBy('assignment_start_month')
   .agg(F.count(F.col('assignment_start_month')).alias('count'))
   .toPandas().set_index('assignment_start_month').sort_index()
   .plot.bar()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## dbutils Functions
# MAGIC What are those `dbutils` commands?  They are special non-spark commands for Databricks for certain operations.
# MAGIC 
# MAGIC Check out the [dbutils filesystem utilites](https://docs.microsoft.com/en-us/azure/databricks/dev-tools/databricks-utils#--file-system-utility-dbutilsfs) page for a deeper explination, but the ones you'll use most are these...
# MAGIC - `dbutils.fs.ls` - Just like `ls` on a unix file system but it should be able to span any cloud resource you have access to.  This is a great way to check if a file exist and if you have permission to that file.
# MAGIC - `dbutils.fs.rm` - Like `rm` on a unix system, it will delete one file or recursively delete a directory.
# MAGIC - `dbutils.fs.put` - A quick an dirty way to put a text string into a cloud location; think of it like `touch` or `cat`
# MAGIC 
# MAGIC Others like `mv` and `cp` may be useful but mileage may vary from system to system.
# MAGIC 
# MAGIC **NOTE**: One command family that is discouraged is `mount` and `unmount`.  These commands may mount a cloud location for an entire cluster and may transmit permissions of the mounter to any of users.  Namely this would open up a cloud location with your credentials but anyone on this Databricks cluster could read or write as you.

# COMMAND ----------

# MAGIC %md 
# MAGIC # Writing to your Scratch
# MAGIC Did you do it? Your very own scratch space? We crafted a few short cuts that may help to write to a user's home directory without having to hard code a name and/or update constant ahead of time.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Checking your scratch
# MAGIC Let's test the file to see if it exists first, using `ls` and `put` as described above.

# COMMAND ----------

import datetime as dt

try:
    list_scratch = dbutils.fs.ls(SCATCH_ROOT)
    fn_log(f"## Files in user scratch ... {len(list_scratch)} total from path '{SCATCH_ROOT}'")
    for file_ref in list_scratch[:5]:   # just pring the first five
        fn_log(f"{file_ref.name}: {file_ref.size} bytes")
    fn_log("")
except Exception as e:
    fn_log(f"Your scratch may not exist ... '{e}'")

try:
    dbutils.fs.put(f"{SCATCH_ROOT}/test-file.txt", f"File updated... {dt.datetime.now()}", True)
except Exception as e:
    fn_log(f"Failed to create a place holder file, does your scratch exist? ... '{e}'")

fn_log(f"\nIf everything worked, you should be able to open this url (you'll need to copy/paste)....\n\n{SCRATCH_URL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Writing to a CSV
# MAGIC Generally this should be avoided  -- because we don't want to locally download or share data from a cloud source -- but the request is frequent enough that it might be helpful.  If you need more of a walkthrough consider [this single CSV reference](https://sparkbyexamples.com/spark/spark-write-dataframe-single-csv-file/).

# COMMAND ----------

import numpy as np
import pandas as pd
df_random = pd.DataFrame(np.random.rand(100,4), columns=[f"rand{i:02}" for i in range(4)])
sdf_random = spark.createDataFrame(df_random)
display(sdf_random)

def write_csv_single(sdf, path_target):
    """A utility function to write a spark data frame (`sdf`) to a single CSV destination (`path_target`)..."""
    try:
        dbutils.fs.rm(path_target, recurse=True)
    except Exception as e:
        pass
    # the main trick is to reparition to just `1` file...
    sdf_random.repartition(1).write.format('csv').option('overwrite', True).option('header', True).save(path_target)
    # find what else went there, looking for the one parquet split...
    stat_file = [(fs.name, fs.size) for fs in dbutils.fs.ls(path_target) if fs.name.endswith('.csv')][0]
    # move newly written to temp file
    path_temp = f"{path_target}.temp"
    dbutils.fs.mv(f"{path_target}/{stat_file[0]}", path_temp)
    # clear just written hierarchy
    dbutils.fs.rm(path_target, recurse=True)
    # move temp file to single location
    dbutils.fs.mv(path_temp, path_target)
    fn_log(f"Wrote to CSV ... {path_target} ({stat_file[1]} bytes), check out the URL above for reference.")
    

path_csv_example = f"{SCATCH_ROOT}/mlworkshop2021_example.csv"
write_csv_single(sdf_random, path_csv_example)

    

# COMMAND ----------

# MAGIC %md
# MAGIC Thanks for walking through this intro to data writes.  We demonstrated how to read and write to a personal scratch space with a few tips on developing optimal indexing when you write and creating individual CSVs.
# MAGIC 
# MAGIC When you're ready, head on to the next script `1c_COMPUTING_FEATURES` where we begin to create a feature pipeline in spark.

# COMMAND ----------

# MAGIC %md
# MAGIC # Done with Writes!
# MAGIC Still want more or have questions about more advanced topics?  Check out the directory `extra_credit` to find a few different notebooks that have more useful details.  Additionally, there are a lot of [advanced features for delta format](https://databricks.com/blog/2019/02/04/introducing-delta-time-travel-for-large-scale-data-lakes.html) that we don't dive into in this workshop.
# MAGIC 
# MAGIC - `1_DATA_IN_DEEP` - Examples of getting a token from DEEP, storing personal secrets, reading from DEEP, writing to DEEP.  This extra ccredit script is definitely not required and is often referred to as a POC (proof of concept) environment instead of a business-sustained platform.
