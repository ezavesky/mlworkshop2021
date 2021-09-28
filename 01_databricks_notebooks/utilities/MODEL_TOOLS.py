# Databricks notebook source
# MAGIC %run ./WORKSHOP_CONSTANTS

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, LinearSVC
from pyspark.ml.tuning import ParamGridBuilder

# simple function to get a raw clasifier
def create_untrained_classifier(classifier, col_features, param_grid=False):
    """
    Return a classifier and some parameters to search over for training. 
    Want to see the orgin of this function?  Head to workbook 1d for some early training lessons
    """
    # https://spark.apache.org/docs/latest/ml-classification-regression.html
    if classifier == "RF":
        # https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.RandomForestClassifier.html?highlight=randomforestclassifier#pyspark.ml.classification.RandomForestClassifier
        cf = RandomForestClassifier(featuresCol=col_features, labelCol=IHX_COL_LABEL,
                            predictionCol=IHX_COL_PREDICT_BASE.format(base="int"),
                            probabilityCol=IHX_COL_PREDICT_BASE.format(base="prob"),
                            rawPredictionCol=IHX_COL_PREDICT_BASE.format(base="raw"))
        cf.setNumTrees(100)
        cf.setMaxDepth(10)
        grid = (ParamGridBuilder()   # this "grid" specifies the same settings but for a different funcction
            .addGrid(cf.numTrees, [20, 50, 100] if param_grid else [cf.getNumTrees()])
            .addGrid(cf.maxDepth, [5, 10, 20] if param_grid else [cf.getMaxDepth()])
            .build()
        )
    elif classifier == "SVM":
        # https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.LinearSVC.html#pyspark.ml.classification.LinearSVC
        cf = LinearSVC(featuresCol=col_features, labelCol=IHX_COL_LABEL,
                        predictionCol=IHX_COL_PREDICT_BASE.format(base="int"),
                        # probabilityCol=IHX_COL_PREDICT_BASE.format(base="prob"),
                        rawPredictionCol=IHX_COL_PREDICT_BASE.format(base="prob"))
        cf.setMaxIter(15)
        cf.setRegParam(0.1)
        grid = (ParamGridBuilder()   # this "grid" specifies the same settings but for a different funcction
            .addGrid(cf.maxIter, [10, 15, 20] if param_grid else [cf.getMaxIter()])
            .addGrid(cf.regParam, [0.001, 0.01, 0.1] if param_grid else [cf.getRegParam()])
            .build()
        )
    elif classifier == "LR":
        cf = LogisticRegression(featuresCol=col_features, labelCol=IHX_COL_LABEL,
                                predictionCol=IHX_COL_PREDICT_BASE.format(base="int"),
                                probabilityCol=IHX_COL_PREDICT_BASE.format(base="prob"),
                                rawPredictionCol=IHX_COL_PREDICT_BASE.format(base="raw"))
        # We can also use the multinomial family for binary classification
        cf.setMaxIter(15)
        cf.setTol(1E-6)
        cf.setRegParam(0.1)
        cf.setElasticNetParam(0.8)
        cf.setFamily("multinomial")
        grid = (ParamGridBuilder()   # this "grid" specifies the same settings but for a different funcction
            .addGrid(cf.maxIter, [10, 15] if param_grid else [cf.getMaxIter()])
            .addGrid(cf.tol, [1e-6, 1e-8] if param_grid else [cf.getTol()])
            .addGrid(cf.regParam, [0.001, 0.01, 0.1] if param_grid else [cf.getRegParam()])
            .addGrid(cf.elasticNetParam, [0.8, 0.75] if param_grid else [cf.getElasticNetParam()])
            .build()
        )

    return cf, grid

# COMMAND ----------

# example function for converting tbeween params to hyperopt

def param_grid_to_hyperopt(grid):
    """
    Utility to convert from grid parameters to hyperopt parameters.
    """
    from hyperopt import hp
    dict_param = {}
    for o in grid:
        for j in o:
            if j.name not in dict_param:
                dict_param[j.name] = []
            dict_param[j.name].append(o[j])
    # print(dict_param)
    dict_param = {k: hp.choice(k, list(set(dict_param[k]))) for k in dict_param}
    # print(dict_param)
    return dict_param
