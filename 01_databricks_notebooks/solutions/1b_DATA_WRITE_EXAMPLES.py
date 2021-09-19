# Databricks notebook source
# MAGIC %md
# MAGIC # Part 1 - MLWorkshop2021
# MAGIC This noteobook is part of the [2021 ML Workshop](https://INFO_SITE/cdo/events/internal-events/90dea45e-1454-11ec-8dca-7d45a6b8dd2a) (sponsored by the Software Symposium)

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

# MAGIC %run ./0_WORKSHOP_CONSTANTS

# COMMAND ----------

# MAGIC %md
# MAGIC ## Writing a delta dataframe

# COMMAND ----------

# example of loading CSV and writing to a ADLSg2 location in delta format
sdf_ihx = spark.read.format('csv').option('header', True).load(f"{MLW_DATA_ROOT}/ihx/IHX-training.csv")

# you may need to delete a prior version because delta will otherwise optimize it as an update
try:
    dbutils.fs.rm(IHX_GOLD, True)   # recursively delete what was there before
except Excpetion as e:
    print("Error clearing parititon, maybe it didn't exist...")

# NOTE: You may not be able to run the first command this due to privilages, but if you've created a 
# scratch space under your user ID, the seccond command should work.
sdf_ihx.repartition('region').write.format('delta').partitionBy('region').mode('overwrite').save(f"{IHX_GOLD}")


# COMMAND ----------

# print the contents of a directory we'll be using for this workshop
ROOT_DATA = "abfss://mlworkshop2021@STORAGE"
try:
    print(dbutils.fs.ls(ROOT_DATA))
except Exception as e:
    print(f"If you got an error, the most likely problem permissions! '{e}'")

# COMMAND ----------

# MAGIC %md
# MAGIC ### We can write out one single ban Showtime viewership to a storage container we have write permissions to

# COMMAND ----------

## This container we do not have permissions to write in!
try:
    one_ban.write.csv(f'abfss://eview-delta@taapexpstorage.STORAGE/{USER_ID}/one_ban')
except Exception as e:
    print(f"Ooops, you can't write to this location, this error is expected... {e}")

# COMMAND ----------

## This container has write permissions for our group
# Make sure you write out to your own folder though!
# NOTE: we're truncating the write below because CSVs are really big, and you get it anyway, right?
one_ban.write.limit(5000).mode('overwrite').csv(f'abfss://dsair@dsairgeneraleastus2sa.STORAGE/users/{USER_ID}/one_ban')

# COMMAND ----------

## This container has write permissions for our group
# another example of writing into DELTA format
# NOTE: we're truncating the write below because it can be really big , and you get it anyway, right?
one_ban.write.mode('overwrite').format('delta').save(f'abfss://dsair@dsairgeneraleastus2sa.STORAGE/users/{USER_ID}/one_ban_delta')
