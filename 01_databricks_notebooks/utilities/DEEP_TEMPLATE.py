# Databricks notebook source
# MAGIC %md
# MAGIC #### Template to Read/Write Data with DEEP
# MAGIC 1. REST API - create empty dataset (Manually for now)
# MAGIC   - you must create a desination dataset
# MAGIC 2. Spark write data (parquet format)
# MAGIC   - using dataframe-like write, POSTS to DEEP API
# MAGIC 3. REST API - Apply Schema (Manually for now)
# MAGIC   - after evaluation, you'll need to click `Apply Schema`
# MAGIC * DEEP URL: https://paloma.palantirfoundry.com/workspace/slate/documents/deep-foundry

# COMMAND ----------

# Provide your DEEP Token Here
# tokens for access to other systems?

# $ databricks secrets write --scope personal-ID-scope --key deep_token --string-value "TOKEN"

# set spark.databricks.userInfoFunctions.enabled = true;
spark.conf.set("spark.databricks.userInfoFunctions.enabled", "true")
USER_PATH = spark.sql('select current_user() as user;').collect()[0]['user']
USER_ID = USER_PATH.split('@')[0]
print(f"Detected User ID: {USER_ID}")
DEEP_TOKEN = dbutils.secrets.get(scope=f'personal-{USER_ID}-scope', key='deep_token')
print(f"Successfully loaded DEEP token '{DEEP_TOKEN}' (should be redacted)")

# old way, please dont' hard code tokens!!
# DEEP_TOKEN = "TOKEN" 

# spark.conf.set("spark.hadoop.foundry.catalog.bearerToken", DEEP_TOKEN)
FOUNDRY_FQDN = "paloma.palantirfoundry.com"

# COMMAND ----------

# import requests 
# import json

# endpoint = "https://paloma.palantirfoundry.com/foundry-catalog/api/catalog/datasets"
# headers = {"authorization": "Bearer " + DEEP_TOKEN}
# response = requests.get(endpoint, headers=headers)

# parsed = json.loads(response.text)
# print(json.dumps(parsed, indent=4, sort_keys=True))

# COMMAND ----------

# Python Wrapper
from pypalantir_core import FoundryClient
from pypalantir_operations import rid_for_path as getDatasetRid

def loadDataframe(datasetPath):
  client = FoundryClient(host='https://{}'.format(FOUNDRY_FQDN), token=DEEP_TOKEN)
  
  return spark.read.parquet('foundry://{}@{}/datasets/{}/views/master/files/*'.format(DEEP_TOKEN, FOUNDRY_FQDN, getDatasetRid(client, datasetPath)))

def loadDataframeFromCSV(datasetPath):
  client = FoundryClient(host='https://{}'.format(FOUNDRY_FQDN), token=DEEP_TOKEN)
  
  return spark.read.csv('foundry://{}@{}/datasets/{}/views/master/files/*'.format(DEEP_TOKEN, FOUNDRY_FQDN, getDatasetRid(client, datasetPath)), header=True)

# COMMAND ----------

# read latest view a the dataset on branch master
if False:
  df = loadDataframe("/Innovation Lab/Foundry Training and Resources/Reference Examples/Java UDFs/outputData")
  df.limit(5).show()

# COMMAND ----------

# imports
import requests
import json

spark.conf.set("spark.sql.sources.commitProtocolClass", "org.apache.spark.sql.execution.datasources.SQLHadoopMapReduceCommitProtocol")

def getDatasetRid(datasetPath):
  headers = {"Content-Type": "application/json", "Authorization": "Bearer {}".format(DEEP_TOKEN)}
  payload = [datasetPath]
  resp = requests.post("https://{}/compass/api/batch/resources-by-paths".format(FOUNDRY_FQDN), headers=headers, json=payload)
  
  if resp.status_code != 200:
    raise RuntimeError("Failed loading rid for dataset {}.".format(datasetPath))    
  
  result = dict([(key, value["rid"]) for key, value in json.loads(resp.text).items()])
  return result[datasetPath]

def setTransactionType(datasetRid, transactionRid, transactionType):
  headers = {"Content-Type": "application/json", "Authorization": "Bearer {}".format(DEEP_TOKEN)}
  payload = transactionType
  resp = requests.post("https://{}/foundry-catalog/api/catalog/datasets/{}/transactions/{}".format(FOUNDRY_FQDN, datasetRid, transactionRid), headers=headers, json=payload)
  
  if resp.status_code != 200:
    raise RuntimeError("Failed setting transaction type for transaction {} on dataset {} to {}.".format(transactionRid, datasetRid, transactionType))
  
  # return success
  return True

def startTransaction(datasetRid, branch):
  headers = {"Content-Type": "application/json", "Authorization": "Bearer {}".format(DEEP_TOKEN)}
  payload = {"branchId": branch, "record": {}}  
  resp = requests.post("https://{}/foundry-catalog/api/catalog/datasets/{}/transactions".format(FOUNDRY_FQDN, datasetRid), headers=headers, json=payload)
  
  if resp.status_code != 200:
    raise RuntimeError("Failed opening transaction for dataset {} on branch {}.".format(datasetRid, branch))
  
  # return transaction rid
  return json.loads(resp.text)['rid']


def commitTransaction(datasetRid, transactionRid):
  headers = {"Content-Type": "application/json", "Authorization": "Bearer {}".format(DEEP_TOKEN)}
  payload = {"record": {}}  
  resp = requests.post("https://{}/foundry-catalog/api/catalog/datasets/{}/transactions/{}/commit".format(FOUNDRY_FQDN, datasetRid, transactionRid), headers=headers, json=payload)
  
  if resp.status_code != 204:
    raise RuntimeError("Failed committing transaction {} for dataset {}.".format(transactionRid, datasetRid))
  
  # return success
  return True

def writeFiles(datasetRid, transactionRid, df):
  df.write.parquet('foundry://{}@{}/datasets/{}/transactions/{}/spark/'.format(DEEP_TOKEN, FOUNDRY_FQDN, datasetRid, transactionRid))

def loadDataframe(datasetPath):
  rid = getDatasetRid(datasetPath)
  url = 'foundry://{}@{}/datasets/{}/views/master/files/*'.format(DEEP_TOKEN,FOUNDRY_FQDN, rid)
  return spark.read.parquet(url)

def writeDataframe(df, datasetPath, branch="master", mode="SNAPSHOT"):
  # get dataset rid
  datasetRid = getDatasetRid(datasetPath)

  # start transaction
  transactionRid = startTransaction(datasetRid, branch)
  
  # set transaction type
  setTransactionType(datasetRid, transactionRid, mode)
  
  # write files
  writeFiles(datasetRid, transactionRid, df)
  
  # commit transaction
  commitTransaction(datasetRid, transactionRid)
  
  


# COMMAND ----------

# write back to target dataset
# df = <dataframe from somewhere>
# writeDataframe(df, "/Users/cv0361@DOMAIN/SampleData/Dataset - Titanic")
if False:
  writeDataframe(df, "/Innovation Lab/AIaaS DEEP Repository/data/Aurum-test3")


# COMMAND ----------

# writeDataframe(df, "/Innovation Lab/AIaaS DEEP Repository/data/Dataset - Testing")

# COMMAND ----------


