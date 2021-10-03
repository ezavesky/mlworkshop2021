# Databricks notebook source
# MAGIC %md
# MAGIC # Reusing Existing Scripts
# MAGIC If you've attempted to existing source with the "Repositories" function, you may have noticed that some scripts are not included.  Luckily, there's an easy fix for this "feature" which will allow any arbitaray script to be included.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Header Line Simplicity 
# MAGIC In a typical source file, you'd start with importing packages, a legal disclaimer, or other tagging.  For a script to be sync'd and recognized within Databricks, you need to add this line ...
# MAGIC 
# MAGIC ```
# MAGIC # Databricks notebook source
# MAGIC ```
# MAGIC 
# MAGIC Simple, right? If you need another example, checkout the repo containing the first notebook file - [CDO DevOps instance](https://dev.azure.com/dsair/dsair_collaborator/_git/mlworkshop2021?path=/01_databricks_notebooks/solutions/1a_DATA_READ_AND_LIST.py).
