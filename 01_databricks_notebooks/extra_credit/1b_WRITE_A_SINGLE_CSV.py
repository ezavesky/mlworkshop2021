# Databricks notebook source
# MAGIC %md 
# MAGIC # Writing to your Scratch
# MAGIC Did you do it? Your very own scratch space? We crafted a few short cuts that may help to write to a user's home directory without having to hard code a name and/or update constant ahead of time.  
# MAGIC 
# MAGIC If you need to see how to create and experiment with your own scratch head over to work book 1b or jump directly to the  [self service CDO page](https://INFO_SITE/sites/data/SitePages/Creating-an-ADLSGen2-Scratch-Space.aspx) to understand how to get your own slice!

# COMMAND ----------

# MAGIC %run ../utilities/WORKSHOP_CONSTANTS

# COMMAND ----------

# MAGIC %md
# MAGIC ## Checking your scratch
# MAGIC Let's test the file to see if it exists first, using `ls` and `put` as described above.

# COMMAND ----------

import datetime as dt
path_csv_example = f"{SCATCH_ROOT}/mlworkshop2021_example.csv"

try:
    list_scratch = dbutils.fs.ls(SCATCH_ROOT)
    fn_log(f"## Files in user scratch ... {len(list_scratch)} total from path '{SCATCH_ROOT}'")
    for file_ref in list_scratch[:5]:   # just pring the first five
        fn_log(f"{file_ref.name}: {file_ref.size} bytes")
    fn_log("")
except Exception as e:
    fn_log(f"Your scratch may not exist ... '{e}'")

try:
    dbutils.fs.put(path_csv_example, f"File updated... {dt.datetime.now()}", True)
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
