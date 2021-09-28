# Databricks notebook source
# MAGIC %md 
# MAGIC 
# MAGIC # Data in DEEP
# MAGIC DEEP is AT&T's branded name for [Palantir's Foundry Application](https://paloma.palantirfoundry.com/workspace/slate/documents/deep-foundry).  In the words of others DEEP is...
# MAGIC 
# MAGIC ```
# MAGIC ...  it is a prototyping and research capability, not a long-term production platform.  
# MAGIC Many, though important to our business partners, don’t support processes that 
# MAGIC require 99.999% reliability and are classified as non-production.  Those may stay in 
# MAGIC DEEP for the time-being as long as everyone’s expectations on production support 
# MAGIC are aligned...
# MAGIC ```
# MAGIC 
# MAGIC This means that Databricks is still the preferred compute platform and Databricks + Snowflake are the preferred storage environments.  
# MAGIC 
# MAGIC Some businesses have chosen to utilize DEEP for its cost, easy to create visualization / interactions, or hybrid comptue and deployment environments, but generally you should strive to complete your tasks in a combination of Databricks, Snowflake, and maybe h2o (still TBD as of Sept 2021).
# MAGIC 
# MAGIC _(update 08/29/21)_
# MAGIC - You can follow the directions below, which include a template that may simply some interactions or check out another [how-to access guide in CDO reference](https://INFO_SITE/sites/data/SitePages/Access-Data-from-DEEP.aspx).

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Retrieve a DEEP token
# MAGIC * You'll need to retrieve a DEEP token and put it into a Databricks secret (we'll do that below)
# MAGIC * Step zero, you'll need a corporate RSA Token to login, if you don't have one [activate an RSA key](https://AUTH_SITE/iamportal-prod/rsauser/#/home)
# MAGIC * First, head over to the [DEEP/Foundry main page](https://paloma.palantirfoundry.com/workspace/slate/documents/deep-foundry)
# MAGIC * Second, click `Account` -> `Settings` 
# MAGIC * A new page will open and navigate to `Tokens` (on the left) and click the green `Create Token` button
# MAGIC * Enter any name (it's just for tracking), and then immediately COPY and save the token (you'll need it below.)
# MAGIC 
# MAGIC ## Storage of DEEP Token as an Databricks Secret
# MAGIC * Instead of saving a token as a hardcoded entry, consider using Databricks secrets.
# MAGIC * Run though of [CLI secrets workflow](https://wiki.SERVICE_SITE/x/2gpiWQ)
# MAGIC * Use the AIaaS Secret Scoping Tool (say that three times fast, too!)
# MAGIC   * A tool developed by CDO may help to do this secrets management without the CLI 
# MAGIC     * [Secrets Managment Webtool](https://INFO_SITE/sites/data/SitePages/Simplifying-Databricks-Secret-Management.aspx)
# MAGIC * If you've done it right, a command like the one below will store your token  
# MAGIC ```
# MAGIC $ databricks secrets write --scope personal-ATTID-scope --key deep_token --string-value "TOKEN"
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read/Write template
# MAGIC First, let's look at a template to read and write to DEEP.  
# MAGIC - Here's an example file `WALK_06_DEEP-TEMPLATE` (derived from this shared sample [DEEP Template](https://adb-2754134964726624.4.azuredatabricks.net/?o=2754134964726624#notebook/2842883056886338/command/2842883056886339))
# MAGIC - **NOTE** Clusters do require special libraries to run the DEEP connector.  As of 8/26/21, they are as follows and you can copy/paste these URLs into your library reference. (e.g. see )
# MAGIC   - [foundry_fsclient_3_53_0_all.jar](dbfs:/FileStore/jars/1857c41f_e4bb_4785_87a7_e6e2aee3f624-foundry_fsclient_3_53_0_all-6a3e8.jar)
# MAGIC   - [pypalantir_core-0.26.6-py3-none-any.whl](dbfs:/FileStore/jars/6a7d3aa9_f780_462e_a659_ea497901a5de/pypalantir_core-0.26.6-py3-none-any.whl)
# MAGIC   - [pypalantir_operations-0.3.0-py3-none-any.whl](dbfs:/FileStore/jars/aae57883_36c1_4ad8_8e76_5632ae9b7685/pypalantir_operations-0.3.0-py3-none-any.whl)
# MAGIC 
# MAGIC We'll use the "magic command" `%run ../utilities/DEEP-TEMPLATE` to load the script into our environment.  Note that this command is very finicky, so you'll need to use it on a single line and try to not use absolute paths.

# COMMAND ----------

# MAGIC %run ../utilities/DEEP_TEMPLATE

# COMMAND ----------

# https://paloma.palantirfoundry.com/workspace/data-integration/dataset/preview/ri.foundry.main.dataset.86b480e6-a490-4b3a-80e6-53a15ebcbad9/master
df = loadDataframe("/Innovation Lab/Foundry Training and Resources/Reference Examples/Java UDFs/outputData")
df.limit(5).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Operations
# MAGIC Access to data DEEP is tightly controlled by CSO and Governance.  This means it will be hard to get your data "from the wild" into a trusted project.  
# MAGIC 
# MAGIC There are currently two planned methods for writing to DEEP (as of 08/27/21)
# MAGIC 
# MAGIC 1. *USER PUSH MODEL*: via API (see examples in the includes script, we'll try here, too) 
# MAGIC   - this isn't fully approved by governance yet
# MAGIC   - we'll try it anyway, for the purposes of education
# MAGIC 2. *SYSTEM PULL MODEL*: a managed linkage between a storage container and a source associated to a DEEP project --- this requires the help of an admin to link an Azure stoage container (but we reviewed how you can put data there in WALK-02), but ideally you could safely add data and have 
# MAGIC   - this is an active test/PoC that will launch at the start of September; ping DEEP's contact Michael Williams (mwilliams@palantir.com) and the org volunteers if you want to get in on the 'fun'
# MAGIC   - the challenge with this solution is that it has a longer lead/preconfigure, but it may be easier in the long run
# MAGIC   - another challenge, this is under testing is the latency/SLA for auto-ingestion of changed data (Azure -> DEEP)

# COMMAND ----------

# MAGIC %md
# MAGIC - We're going to create a custom dataset in your user's home directory
# MAGIC - First, navigate to your "home" directory from this link [DEEP projects](https://paloma.palantirfoundry.com/workspace/compass/projects) -> then click "Your Files" _(NOTE: This is a unique URL / path for every user)_
# MAGIC - Click the green `New Dataset` button 
# MAGIC - Go back to your user directory and rename the dataset to something reasonable, like "Dataset-WALKTEST"
# MAGIC - Click once on the new dataset and copy the `Location` setting from the pane on the right, SAVE THIS PATH FOR BELOW
# MAGIC   - Also, _don't close this window_, because we'll come back to it to inspect our output below...

# COMMAND ----------

path_write = '/Users/km430p@DOMAIN/km430p_test/Dataset - 2021-08-31 12:43:36'

# COMMAND ----------

# MAGIC %md
# MAGIC - When you're ready, execute the lines below to write our previous data frame to this new location.

# COMMAND ----------

writeDataframe(df, path_write, branch="master")

# COMMAND ----------

# MAGIC %md
# MAGIC - Okay, let's go back to the DEEP window and check out your new dataframe.
# MAGIC - Weird, it looks like a bunch of delta files and no nice table!
# MAGIC - This is a limitation of the current *USER PUSH* method, but all you need to do is click `Apply Schema`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using Outside Data in a DEEP Project
# MAGIC - The method above is one techncial way to do this, but it may not work for other things like images or binary blobs.
# MAGIC - Additionally, to get your data to be visible inside of a DEEP project (to move it from your personal space to the DEEP project), you have to fill out a DEEP ticket.
# MAGIC   - Start the ticket from the dataset you're working with
# MAGIC   - Indicate which project the data should be going to
# MAGIC   - Optionally, tag DEEP team members who are working on the project with you
# MAGIC   - Here's [an example ticket](https://paloma.palantirfoundry.com/workspace/issues-app/issue/ri.issues.main.issue.ddabc48a-e5ae-48e8-9b61-e666f312879f) for reference, but don't worry if you can't read this one, it just reiterates the above items.
# MAGIC - We hope to have a more automated way to get data in, but honestly your the DEEP employees that you interface should be able to get most of the company standard / required sources imported for you.
