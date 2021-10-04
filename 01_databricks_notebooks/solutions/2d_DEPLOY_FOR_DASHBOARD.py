# Databricks notebook source
# MAGIC %md
# MAGIC # Part 2 - MLWorkshop2021
# MAGIC This notebook is part of the [2021 ML Workshop](https://INFO_SITE/cdo/events/internal-events/90dea45e-1454-11ec-8dca-7d45a6b8dd2a) (sponsored by the Software Symposium)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Clone This Demo Notebook
# MAGIC 1. Head to http://FORWARD_SITE/cdo-databricks  (don't worry, it's a fowarding DNS to the main CDO PROD(uction) instance)
# MAGIC 2. Log-in via your AT&T ID and DOMAIN password
# MAGIC 3. Navigate to this script (Repos -> ez2685@DOMAIN -> mlworkshop2021 -> 01_databricks_notebooks -> (solutions) -> `2d_DEPLOY_FOR_DASHBOARD`)
# MAGIC 4. Clone the file (either `File` -> `Clone` or above with right-click)
# MAGIC    - Alternatively, add a repo (left side menu) for the [MLWorkshop2021](https://dsair@dev.azure.com/dsair/dsair_collaborator/_git/mlworkshop2021) that is stored in [Azure DevOps](https://dev.azure.com/dsair/dsair_collaborator/_git/mlworkshop2021) and jump into your own url..
# MAGIC 5. *Pat on back* or *high five*

# COMMAND ----------

# MAGIC %run ../utilities/WORKSHOP_CONSTANTS

# COMMAND ----------

# MAGIC %run ../utilities/MLFLOW_TOOLS

# COMMAND ----------

# MAGIC %run ../utilities/EVALUATOR_TOOLS

# COMMAND ----------

# MAGIC %md
# MAGIC # Final Rendering to Dashboard
# MAGIC In the push to data-driven reporting, we should make it easy to generate and delivery model performance to your stakeholders!  Luckily if you build your model with `mlflow` tracking, it's trivial to query and print the results of each model that you've trained.
# MAGIC 
# MAGIC In the section below, we'll generate a simple plot that aggregates performance of different tests using programmatic access from mlflow.

# COMMAND ----------

# start local experiment
experiment = databricks_mlflow_create(MLFLOW_EXPERIMENT)
fn_log("\n###### MLFLOW EXPERIMENT ##############")
fn_log(experiment)

# query for all runs...
df_runs = mlflow.search_runs([experiment.experiment_id], order_by=["metrics.CG2D DESC", "metrics.avg_CG2D DESC"])
fn_log("\n###### MLFLOW RUNS ##############")
fn_log(df_runs)

# COMMAND ----------

import numpy as np
from matplotlib import pyplot as plt

## PART 1 - query mlflow!
df_runs['metrics.CG2D'] = df_runs.apply(lambda r: r['metrics.avg_CG2D'] if np.isnan(r['metrics.CG2D']) else r['metrics.CG2D'], axis=1)

## PART 2 - normalize names
def _map_run_name(row_data):
    import re
    import textwrap

    re_clean = re.compile(r"^params\.")
    
    name_base = row_data['tags.mlflow.runName']
    dt_format = ""
    if row_data['tags.mlflow.runName'] is None:
        if row_data['params.mlModelClass'] == "RandomForestClassifier":
            name_base = "RF"
        elif row_data['params.mlModelClass'] == "LogisticRegression": 
            name_base = "LR"
        elif row_data['params.mlModelClass'] == "LinearSVC": 
            name_base = "SVM"
        elif row_data['params.mlModelClass'] == "GBTClassifier": 
            name_base = "GBT"
        else:
            name_base = "Unknown"
        dt_format = "-" + row_data['start_time'].strftime('%m%d-%H%M')
    # find okay param keys
    keys_valid = [k for k in list(row_data.index) if "param" in k and "Uid" not in k and "ModelClass" not in k]
    if False:   # rich param info
        # pull the value
        list_res = [f"{re_clean.sub('', k)}:{row_data[k]}" for k in keys_valid if row_data[k] is not None]
        # wrap the individua lines
        line_params = "\n".join(textwrap.wrap(', '.join(list_res), 30))
    else:       # run id
        line_params = f"id: {row_data['run_id'][:8]}"  # truncate to first hash parts
    return f"{name_base}{dt_format}\n{line_params}" # "-{row_data['tags.search-type']}"

# fill-in the run names
df_runs['run_name'] = df_runs.apply(lambda r: _map_run_name(r), axis=1)

## PART 3 - filter for valid runs, grab the top

# filter for those which have non-failed status
df_runs_valid = df_runs[(~df_runs['metrics.CG2D'].isna()) & (df_runs['status'].str.upper() != "FAILED")]
df_runs_trim = df_runs_valid.head(10) # .set_index('run_name')
df_runs_trim = df_runs_trim.sort_values('metrics.CG2D', ascending=True).set_index('run_name')
col_informative = ['run_id', 'tags.mlflow.runName', 'status', 'artifact_uri', 'start_time', 'metrics.CG2D', 'tags.search-type', 'params.mlModelClass'] #  'runSource'] #, 'tags.s.release', 'params.estimator']

## PART 3 - now plot the results

def _plot_top_runs(df):
    fig, ax = plt.subplots(1, 1, figsize=(6,8))
    df['metrics.CG2D'].plot.barh(ax=ax)
    ax.set_xlabel('DCG (2nd Decile)')
    ax.set_ylabel('Run Name')
    ax.grid('both')
    fig.show()
    fig.set_dpi(120.0)
    return fig
    
# fn_log(df_trim.columns.tolist())
_plot_top_runs(df_runs_trim)
display(df_runs_valid[col_informative])


# COMMAND ----------

# MAGIC %md
# MAGIC # Done with Grid Exploration and Logging!
# MAGIC Still want more or have questions about more logging model exploration?  Continue on to either notebook 2b or 2c! 
