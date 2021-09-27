# Databricks notebook source
# This file contains constants used for the workshop!
# You can (and should) change them for your own experiments, but they are uniquely defined
# here for constants that we will use together.

# COMMAND ----------

# our root directory for access in databricks
MLW_DATA_ROOT = "abfss://mlworkshop2021@STORAGE"
# absolute reference for the above storage container
MLW_DATA_URL = f"https://PORTAL/#blade/Microsoft_Azure_Storage/ContainerMenuBlade/overview/storageAccountId/%2Fsubscriptions%2F81b4ec93-f52f-4194-9ad9-57e636bcd0b6%2FresourceGroups%2Fblackbird-prod-storage-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Fblackbirdproddatastore/path/mlworkshop2021/etag/%220x8D9766DE75EA338%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride//defaultId//publicAccessVal/None"

# the "gold" data reference - in CDO parlence, this generally meant the data has been cleaned and indexed optimally
IHX_GOLD = f"{MLW_DATA_ROOT}/ihx/IHX_gold"
IHX_GOLD_TESTING = f"{IHX_GOLD}-testing"

# some feature column names
IHX_COL_VECTORIZED = "vectorized"
IHX_COL_NORMALIZED = "normalized"
IHX_COL_INDEX = "jobid"
IHX_COL_LABEL = "final_response"
IHX_COL_PREDICT_BASE = "predict_{base}"

# now some path definitions for our feature vectorization stages
IHX_VECTORIZER_PATH = f"{MLW_DATA_ROOT}/ihx/IHX_Feature_Vectorizer"
IHX_NORM_L2_PATH = f"{MLW_DATA_ROOT}/ihx/IHX_Feature_L2Normalizer"
IHX_NORM_MINMAX_PATH = f"{MLW_DATA_ROOT}/ihx/IHX_Feature_MinMaxScaler"

# intermediate data from vecctorization
IHX_GOLD_TRANSFORMED = f"{MLW_DATA_ROOT}/ihx/IHX_gold_transformed"
IHX_GOLD_TRANSFORMED_TEST = f"{IHX_GOLD_TRANSFORMED}-test"

# full evaluator variables
IHX_TRAINING_SAMPLE_FRACTION = 0.11   # reduce training set from 429k to about 10k for speed during the workshop

# COMMAND ----------

# MAGIC %r
# MAGIC MLW_DATA_ROOT <- "abfss://mlworkshop2021@STORAGE"
# MAGIC IHX_GOLD <- paste(MLW_DATA_ROOT, "/ihx/IHX_gold")
# MAGIC 
# MAGIC # Getting errors running on another cluster? Consider commenting out this cell...

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
REPO_BASE = f"/Repos/{USER_PATH}"   # this is where the repos may appear by default
MLFLOW_EXPERIMENT = "MLWorkshop2021"   # default experiment name for centralized mlflow tracking

# COMMAND ----------

# the last command assumes you make a scratch directory that you own (or can write to) with your ATTID
# for instructions on how to create your own scratch, head to notebook `1b_DATA_WRITE_EXAMPLES`
SCATCH_ROOT = f"abfss://{USER_ID}@STORAGE"

# feature processing paths
SCRATCH_IHX_VECTORIZER_PATH = f"{SCATCH_ROOT}/ihx/IHX_Feature_Vectorizer"
SCRATCH_IHX_MINMAX_PATH = f"{SCATCH_ROOT}/ihx/IHX_Feature_MinMaxScaler"
SCRATCH_IHX_L2_PATH = f"{SCATCH_ROOT}/ihx/IHX_Feature_L2Normalizer"

# this may not work for sure, but let's try to format an Azure Portal for above...
SCRATCH_URL = f"https://PORTAL/#blade/Microsoft_Azure_Storage/ContainerMenuBlade/overview/storageAccountId/%2Fsubscriptions%2F81b4ec93-f52f-4194-9ad9-57e636bcd0b6%2FresourceGroups%2Fblackbird-prod-storage-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Fblackbirdproddatastore/path/{USER_ID}/etag/%220x8D9766DE75EA338%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride//defaultId//publicAccessVal/None"

# COMMAND ----------

# In this section, we define a custom "print" function that can log to python logger or the notebook console
import logging
import os
IS_DATABRICKS = "DATABRICKS_RUNTIME_VERSION" in os.environ

# from pyspark.context import SparkContext
# from pyspark.sql.session import SparkSession

try:
    if logger is None:
        pass
except Exception as e:
    logger = None

if logger is None:
    logger = logging.getLogger(__name__)   
    if not IS_DATABRICKS:
        from .core_constants import CREDENTIALS
    else:
        from sys import stderr, stdout
        # writing to stdout
        handler = logging.StreamHandler(stdout)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
    
        # See this thread for logging in databricks - https://stackoverflow.com/a/34683626; https://github.com/mspnp/spark-monitoring/issues/50#issuecomment-533778984
        log4jLogger = spark._jvm.org.apache.log4j
#     logger = log4jLogger.Logger.getLogger(__name__)  # --- disabled for whitelisting in 8.3 ML 

# also truncate GIT errors
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

def fn_log(str_print):   # place holder for logging
    logger.info(str_print)
    print(str_print)

def is_workshop_admin():
    return USER_ID in ['ez2685']

# COMMAND ----------

def quiet_delete(path_storage):
    try:
        dbutils.fs.rm(path_storage, True)   # recursively delete what was there before
    except Excpetion as e:
        fn_log(f"Error clearing parititon '{path_storage}', maybe it didn't exist...")
