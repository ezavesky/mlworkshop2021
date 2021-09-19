# Databricks notebook source
# This file contains constants used for the workshop!
# You can (and should) change them for your own experiments, but they are uniquely defined
# here for constants that we will use together.

# COMMAND ----------

# our root directory for access in databricks
MLW_DATA_ROOT = "abfss://mlworkshop2021@STORAGE"

# the "gold" data reference - in CDO parlence, this generally meant the data has been cleaned and indexed optimally
IHX_GOLD = f"{MLW_DATA_ROOT}/ihx/IHX_gold"

# COMMAND ----------

# MAGIC %r
# MAGIC MLW_DATA_ROOT <- "abfss://mlworkshop2021@STORAGE"
# MAGIC IHX_GOLD <- paste(MLW_DATA_ROOT, "/ihx/IHX_gold")

# COMMAND ----------

# This is utility code to detect the current user.  It works on the AIaaS cluster to detecct the user ATTID
# which is often the defauly location for new notebooks, repos, etc.

# set spark.databricks.userInfoFunctions.enabled = true;
spark.conf.set("spark.databricks.userInfoFunctions.enabled", "true")
USER_PATH = spark.sql('select current_user() as user;').collect()[0]['user']  # full ATTID@DOMAIN (for directories)
USER_ID = USER_PATH.split('@')[0]   # just ATTID
# print(f"Detected User ID: {USER_ID}")

NOTEBOOK_BASE = f"/Users/{USER_PATH}"   # this is where the experiments will appear in your workspace
EXPERIMENT_BASE = f"/user/{USER_ID}/experiments"   # this is a mounted file path that will store spark artifacts (an adopted constant for AIaaS)
REPO_BASE = f"/Repos/{USER_PATH}"   # this is where the repos may appear by defauly
