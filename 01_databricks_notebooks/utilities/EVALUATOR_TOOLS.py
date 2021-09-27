# Databricks notebook source
# MAGIC %run ./WORKSHOP_CONSTANTS

# COMMAND ----------

# MAGIC %run ./CUMULATIVE_GAIN_SCORE

# COMMAND ----------

from pyspark.ml.evaluation import RankingEvaluator, BinaryClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.functions import vector_to_array

# get the distributed evaluator type from spark for evaluation
def evaluator_obj_get(type_match='CGD'):
    """
    Return a spark performance evaluator.  This is a utility function that prepares an evaluator with
    known column names for the IHX experiment, so column renaming may be necessary elsewhere.
    """
    if type_match == "mAP":
        evalObj = RankingEvaluator(predictionCol=IHX_COL_PREDICT_BASE.format(base="score"),
                                   labelCol=IHX_COL_PREDICT_BASE.format(base="label"), 
                                   metricName='meanAveragePrecision', k=100)
    elif type_match == "AUC":
        # wrap_value = F.udf(lambda v: [ [ v ] ], T.ArrayType(T.FloatType()))
        evalObj = BinaryClassificationEvaluator(rawPredictionCol=IHX_COL_PREDICT_BASE.format(base="int"),
                                                labelCol=IHX_COL_LABEL, metricName='areaUnderROC')
    else:
        evalObj = CuluativeGainEvaluator(rawPredictionCol=IHX_COL_PREDICT_BASE.format(base="prob"), 
                                         labelCol=IHX_COL_LABEL, metricName='CG2D')
    return evalObj


# COMMAND ----------


# # let's also get a confusion matrix...
def evaluator_performance_curve(sdf_predict, str_title=None):
    """
    This function will generate either a precision recall curve or a detection performance curve
    using sci-kit learn's built-ins.  Databricks' version lags the release schedule of scikit 
    so we will dynamically switch to generate the available visual.
    """
    from sklearn import __version__ as sk_versions
    ver_parts = sk_versions.split('.')
    is_new_sk = int(ver_parts[0]) >= 1 or int(ver_parts[1]) >= 24
    df_predict = (sdf_predict
        .withColumn('score', udf_last_prediction(F.col(IHX_COL_PREDICT_BASE.format(base="prob"))))
        .select('score', F.col(IHX_COL_LABEL).alias('label'))
        .toPandas()
    )
    # if we were looking at a cclassificcation problem....
    # from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    # df_predict = sdf_predict.select(IHX_COL_PREDICT_BASE.format(base="int"), IHX_COL_LABEL).toPandas()
    # cm = confusion_matrix(df_predict[IHX_COL_LABEL], df_predict[IHX_COL_PREDICT_BASE.format(base="int")])
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["declined", "engaged"])

    if not is_new_sk:   # scikit v0.24 yet?
        from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
        precision, recall, _ = precision_recall_curve(df_predict['label'], df_predict['score'])
        disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    else:
        from sklearn.metrics import DetCurveDisplay
        disp = DetCurveDisplay.from_predictions(df_predict['score'], df_predict['label']) 
    #                                             display_labels=["declined", "engaged"])
    disp.plot()
    disp.ax_.grid()
    if str_title is not None:  # add title if we need to
        disp.ax_.set_title(str_title)
    disp.figure_.set_dpi(120.0)
    return disp.figure_


# COMMAND ----------


