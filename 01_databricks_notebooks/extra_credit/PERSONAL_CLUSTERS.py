# Databricks notebook source
# MAGIC %md
# MAGIC # Create Personal Cluster 
# MAGIC It may change, but for now, CDO allows people to run in a personal cluster (a simple 2-3 node cluster) for most operations that require lots of special libraries or quick, exclusive operations.  For the vast majority of cases, you can hop on a large shared cluster and run just fine!  However, sometimes special libraries or experimental clusters may be necessary to test the mettle of your code!

# COMMAND ----------

# MAGIC %md
# MAGIC ## Personal Cluster Types 
# MAGIC Choosing a cluster...  
# MAGIC * *PERSONAL* - A personal cluster to install your own packages and run your own tools; lower powered than some org or group operations, but you control it's packages and you're the only user.
# MAGIC * *ORG or GROUP* - If you have a slightly larger job look for those that are associated with CDO or your cluster admin group (e.g. finance, DSAIR, a project-based one, etc).  
# MAGIC   * However, you may also consider `CDO-CEREBRO1-HC` which has is a _"high concurrency"_ cluster and offers some improved spark concurrency.
# MAGIC   * If you're interested in more information, check out this [github discussion](https://github.com/Azure/AzureDatabricksBestPractices/blob/master/toc.md#Deploying-Applications-on-ADB-Guidelines-for-Selecting-Sizing-and-Optimizing-Clusters-Performance) or this [blog entry](https://medium.com/microsoftazure/azure-databricks-workload-types-data-analytics-data-engineering-and-data-engineering-light-b1bb6d36c38c)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Creation Procedure
# MAGIC **TLDR:** Today you'll create a personal cluster for the most flexibility in configuration.
# MAGIC 1. Click on the "Clusters" icon on the left
# MAGIC 2. Click the blue "Create Cluster" button near the top
# MAGIC 3. Select _unrestricted_ policy, use a name like your ID, choose _standard_ mode.
# MAGIC   * The cluster will only be visible to you, but choosing something memorable and distinct may help in the future.
# MAGIC 4. For the "Databricks Runtime Version" it's been suggested to use at least "8.3 XX" where XX is either ML or ML+GPU.  You may have to click the "more" link in the list to get there.
# MAGIC   * _"ML"_ implies machine learning and will have packages like `tensorflow` and `mlflow`.  _"GPU"_ implies a GPU instance will be created.
# MAGIC   * Version "8.3 XX" is suggested because it is well supported and is known to work with [DEEP](https://paloma.palantirfoundry.com/workspace)
# MAGIC 5. When you're all done, click "Create cluster" again and a new cluster will spin up.
# MAGIC   * There are advanced options like adding custom libraries or packages, but we won't need that today.
# MAGIC 6. *Pat on back* or *high five*
# MAGIC 
# MAGIC _(update 08/29/21)_
# MAGIC - You can follow the directions above, which have been evaluated by the DSAIR team or use this [how-to create a personal cluster guide in CDO reference](https://INFO_SITE/sites/data/SitePages/Creating-a-Databricks-Cluster.aspx).
