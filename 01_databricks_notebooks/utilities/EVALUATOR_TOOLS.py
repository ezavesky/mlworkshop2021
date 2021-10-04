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
    from sklearn.metrics import roc_curve, auc
    
    ver_parts = sk_versions.split('.')
    is_new_sk = int(ver_parts[0]) >= 1 or int(ver_parts[1]) >= 24
    df_predict = (sdf_predict
        .withColumn('score', udf_last_prediction(F.col(IHX_COL_PREDICT_BASE.format(base="prob"))))
        .select('score', F.col(IHX_COL_LABEL).alias('label'))
        .toPandas()
    )
    
    fig, ax = plt.subplots(1, 2, figsize=(10,4))
        
    # if we were looking at a cclassification problem....
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
    disp.plot(ax=ax[0])
    disp.ax_.grid()
    if str_title is not None:  # add title if we need to
        disp.ax_.set_title(str_title)
    disp.figure_.set_dpi(120.0)
    
    # now grab other retrieval stats for more interesting display...
    fpr, tpr, _ = roc_curve(df_predict['label'].ravel(), df_predict['score'].ravel())
    roc_auc = auc(fpr, tpr)
    ax[1].plot(fpr, tpr, color='darkorange',
             label='ROC curve (area = %0.2f)' % roc_auc)
    ax[1].plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax[1].set_xlim([0.0, 1.0])
    ax[1].set_ylim([0.0, 1.05])
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_title(f'Receiver operating characteristic (AUC {roc_auc:0.3f})')
    ax[1].legend(loc="lower right")
    ax[1].grid('both')

    return fig 


# COMMAND ----------

# MAGIC %python
# MAGIC 
# MAGIC ### EXPERIMENTAL / DISABLED CODE 
# MAGIC # We need some special plot functions but they aren't yet available in the sklearn version
# MAGIC # associated with this cluster / image.
# MAGIC # So, we consider replicating them in the code below.  At time of writing though, these 
# MAGIC # Functions are disabled because they have not received sufficient testing.
# MAGIC 
# MAGIC # import matplotlib.pyplot as plt
# MAGIC # import scipy as sp
# MAGIC # import numpy as np
# MAGIC 
# MAGIC   
# MAGIC # def plot_DET_curve(fpr, fnr, ax=None, *, title="DET Curve", pos_label=1, **kwargs):
# MAGIC #     """Plot visualization.
# MAGIC #     Parameters
# MAGIC #     ----------
# MAGIC #     ax : matplotlib axes, default=None
# MAGIC #         Axes object to plot on. If `None`, a new figure and axes is
# MAGIC #         created.
# MAGIC #     name : str, default=None
# MAGIC #         Name of DET curve for labeling. If `None`, use `estimator_name` if
# MAGIC #         it is not `None`, otherwise no labeling is shown.
# MAGIC #     **kwargs : dict
# MAGIC #         Additional keywords arguments passed to matplotlib `plot` function.
# MAGIC #     Returns
# MAGIC #     -------
# MAGIC #     display : :class:`~sklearn.metrics.plot.DetCurveDisplay`
# MAGIC #         Object that stores computed values.
# MAGIC #     """
# MAGIC #     # code borrowed from release version of sklearn...
# MAGIC #     #  https://github.com/scikit-learn/scikit-learn/blob/844b4be24/sklearn/metrics/_plot/det_curve.py#L190
# MAGIC 
# MAGIC 
# MAGIC #     if ax is None:
# MAGIC #         _, ax = plt.subplots()
# MAGIC 
# MAGIC #     (line_,) = ax.plot(
# MAGIC #         sp.stats.norm.ppf(fpr),
# MAGIC #         sp.stats.norm.ppf(fnr),
# MAGIC #     )
# MAGIC #     info_pos_label = (
# MAGIC #         f" (Positive label: {pos_label})" if pos_label is not None else ""
# MAGIC #     )
# MAGIC 
# MAGIC #     xlabel = "False Positive Rate" + info_pos_label
# MAGIC #     ylabel = "False Negative Rate" + info_pos_label
# MAGIC #     ax.set(xlabel=xlabel, ylabel=ylabel)
# MAGIC 
# MAGIC #     ax.legend(loc="lower right")
# MAGIC 
# MAGIC #     ticks = [0.001, 0.01, 0.05, 0.20, 0.5, 0.80, 0.95, 0.99, 0.999]
# MAGIC #     tick_locations = sp.stats.norm.ppf(ticks)
# MAGIC #     tick_labels = [
# MAGIC #         "{:.0%}".format(s) if (100 * s).is_integer() else "{:.1%}".format(s)
# MAGIC #         for s in ticks
# MAGIC #     ]
# MAGIC #     ax.set_xticks(tick_locations)
# MAGIC #     ax.set_xticklabels(tick_labels)
# MAGIC #     ax.set_xlim(-3, 3)
# MAGIC #     ax.set_yticks(tick_locations)
# MAGIC #     ax.set_yticklabels(tick_labels)
# MAGIC #     ax.set_ylim(-3, 3)
# MAGIC #     ax.grid('both')
# MAGIC 
# MAGIC #     ax_ = ax
# MAGIC #     figure_ = ax.figure
# MAGIC #     return figure_
# MAGIC 
# MAGIC # def det_curve(y_true, y_score, pos_label=None, sample_weight=None):
# MAGIC #     """Compute error rates for different probability thresholds.
# MAGIC #     .. note::
# MAGIC #        This metric is used for evaluation of ranking and error tradeoffs of
# MAGIC #        a binary classification task.
# MAGIC #     Read more in the :ref:`User Guide <det_curve>`.
# MAGIC #     .. versionadded:: 0.24
# MAGIC #     Parameters
# MAGIC #     ----------
# MAGIC #     y_true : ndarray of shape (n_samples,)
# MAGIC #         True binary labels. If labels are not either {-1, 1} or {0, 1}, then
# MAGIC #         pos_label should be explicitly given.
# MAGIC #     y_score : ndarray of shape of (n_samples,)
# MAGIC #         Target scores, can either be probability estimates of the positive
# MAGIC #         class, confidence values, or non-thresholded measure of decisions
# MAGIC #         (as returned by "decision_function" on some classifiers).
# MAGIC #     pos_label : int or str, default=None
# MAGIC #         The label of the positive class.
# MAGIC #         When ``pos_label=None``, if `y_true` is in {-1, 1} or {0, 1},
# MAGIC #         ``pos_label`` is set to 1, otherwise an error will be raised.
# MAGIC #     sample_weight : array-like of shape (n_samples,), default=None
# MAGIC #         Sample weights.
# MAGIC #     Returns
# MAGIC #     -------
# MAGIC #     fpr : ndarray of shape (n_thresholds,)
# MAGIC #         False positive rate (FPR) such that element i is the false positive
# MAGIC #         rate of predictions with score >= thresholds[i]. This is occasionally
# MAGIC #         referred to as false acceptance propability or fall-out.
# MAGIC #     fnr : ndarray of shape (n_thresholds,)
# MAGIC #         False negative rate (FNR) such that element i is the false negative
# MAGIC #         rate of predictions with score >= thresholds[i]. This is occasionally
# MAGIC #         referred to as false rejection or miss rate.
# MAGIC #     thresholds : ndarray of shape (n_thresholds,)
# MAGIC #         Decreasing score values.
# MAGIC #     See Also
# MAGIC #     --------
# MAGIC #     DetCurveDisplay.from_estimator : Plot DET curve given an estimator and
# MAGIC #         some data.
# MAGIC #     DetCurveDisplay.from_predictions : Plot DET curve given the true and
# MAGIC #         predicted labels.
# MAGIC #     DetCurveDisplay : DET curve visualization.
# MAGIC #     roc_curve : Compute Receiver operating characteristic (ROC) curve.
# MAGIC #     precision_recall_curve : Compute precision-recall curve.
# MAGIC #     Examples
# MAGIC #     --------
# MAGIC #     >>> import numpy as np
# MAGIC #     >>> from sklearn.metrics import det_curve
# MAGIC #     >>> y_true = np.array([0, 0, 1, 1])
# MAGIC #     >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
# MAGIC #     >>> fpr, fnr, thresholds = det_curve(y_true, y_scores)
# MAGIC #     >>> fpr
# MAGIC #     array([0.5, 0.5, 0. ])
# MAGIC #     >>> fnr
# MAGIC #     array([0. , 0.5, 0.5])
# MAGIC #     >>> thresholds
# MAGIC #     array([0.35, 0.4 , 0.8 ])
# MAGIC #     """
# MAGIC #     # link to recent scikit learn function...
# MAGIC #     #  https://github.com/scikit-learn/scikit-learn/blob/844b4be24/sklearn/metrics/_ranking.py#L237
# MAGIC #     fps, tps, thresholds = _binary_clf_curve(
# MAGIC #         y_true, y_score, pos_label=pos_label, sample_weight=sample_weight
# MAGIC #     )
# MAGIC 
# MAGIC #     if len(np.unique(y_true)) != 2:
# MAGIC #         raise ValueError(
# MAGIC #             "Only one class present in y_true. Detection error "
# MAGIC #             "tradeoff curve is not defined in that case."
# MAGIC #         )
# MAGIC 
# MAGIC #     fns = tps[-1] - tps
# MAGIC #     p_count = tps[-1]
# MAGIC #     n_count = fps[-1]
# MAGIC 
# MAGIC #     # start with false positives zero
# MAGIC #     first_ind = (
# MAGIC #         fps.searchsorted(fps[0], side="right") - 1
# MAGIC #         if fps.searchsorted(fps[0], side="right") > 0
# MAGIC #         else None
# MAGIC #     )
# MAGIC #     # stop with false negatives zero
# MAGIC #     last_ind = tps.searchsorted(tps[-1]) + 1
# MAGIC #     sl = slice(first_ind, last_ind)
# MAGIC 
# MAGIC #     # reverse the output such that list of false positives is decreasing
# MAGIC #     return (fps[sl][::-1] / n_count, fns[sl][::-1] / p_count, thresholds[sl][::-1])
# MAGIC 
# MAGIC 
# MAGIC # def _binary_clf_curve(y_true, y_score, pos_label=None, sample_weight=None):
# MAGIC #     """Calculate true and false positives per binary classification threshold.
# MAGIC #     Parameters
# MAGIC #     ----------
# MAGIC #     y_true : ndarray of shape (n_samples,)
# MAGIC #         True targets of binary classification.
# MAGIC #     y_score : ndarray of shape (n_samples,)
# MAGIC #         Estimated probabilities or output of a decision function.
# MAGIC #     pos_label : int or str, default=None
# MAGIC #         The label of the positive class.
# MAGIC #     sample_weight : array-like of shape (n_samples,), default=None
# MAGIC #         Sample weights.
# MAGIC #     Returns
# MAGIC #     -------
# MAGIC #     fps : ndarray of shape (n_thresholds,)
# MAGIC #         A count of false positives, at index i being the number of negative
# MAGIC #         samples assigned a score >= thresholds[i]. The total number of
# MAGIC #         negative samples is equal to fps[-1] (thus true negatives are given by
# MAGIC #         fps[-1] - fps).
# MAGIC #     tps : ndarray of shape (n_thresholds,)
# MAGIC #         An increasing count of true positives, at index i being the number
# MAGIC #         of positive samples assigned a score >= thresholds[i]. The total
# MAGIC #         number of positive samples is equal to tps[-1] (thus false negatives
# MAGIC #         are given by tps[-1] - tps).
# MAGIC #     thresholds : ndarray of shape (n_thresholds,)
# MAGIC #         Decreasing score values.
# MAGIC #     """
# MAGIC #     # link to recent scikit learn function...
# MAGIC #     #  https://github.com/scikit-learn/scikit-learn/blob/844b4be24/sklearn/metrics/_ranking.py#L237
# MAGIC 
# MAGIC #     # Filter out zero-weighted samples, as they should not impact the result
# MAGIC #     if sample_weight is not None:
# MAGIC #         sample_weight = _check_sample_weight(sample_weight, y_true)
# MAGIC #         nonzero_weight_mask = sample_weight != 0
# MAGIC #         y_true = y_true[nonzero_weight_mask]
# MAGIC #         y_score = y_score[nonzero_weight_mask]
# MAGIC #         sample_weight = sample_weight[nonzero_weight_mask]
# MAGIC 
# MAGIC #     # pos_label = _check_pos_label_consistency(pos_label, y_true)
# MAGIC 
# MAGIC #     # make y_true a boolean vector
# MAGIC #     y_true = y_true == pos_label
# MAGIC 
# MAGIC #     # sort scores and corresponding truth values
# MAGIC #     desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
# MAGIC #     y_score = y_score[desc_score_indices]
# MAGIC #     y_true = y_true[desc_score_indices]
# MAGIC #     if sample_weight is not None:
# MAGIC #         weight = sample_weight[desc_score_indices]
# MAGIC #     else:
# MAGIC #         weight = 1.0
# MAGIC 
# MAGIC #     # y_score typically has many tied values. Here we extract
# MAGIC #     # the indices associated with the distinct values. We also
# MAGIC #     # concatenate a value for the end of the curve.
# MAGIC #     distinct_value_indices = np.where(np.diff(y_score))[0]
# MAGIC #     threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
# MAGIC 
# MAGIC #     # accumulate the true positives with decreasing threshold
# MAGIC #     tps = stable_cumsum(y_true * weight)[threshold_idxs]
# MAGIC #     if sample_weight is not None:
# MAGIC #         # express fps as a cumsum to ensure fps is increasing even in
# MAGIC #         # the presence of floating point errors
# MAGIC #         fps = stable_cumsum((1 - y_true) * weight)[threshold_idxs]
# MAGIC #     else:
# MAGIC #         fps = 1 + threshold_idxs - tps
# MAGIC #     return fps, tps, y_score[threshold_idxs]
# MAGIC 
# MAGIC # def stable_cumsum(arr, axis=None, rtol=1e-05, atol=1e-08):
# MAGIC #     """Use high precision for cumsum and check that final value matches sum.
# MAGIC #     Parameters
# MAGIC #     ----------
# MAGIC #     arr : array-like
# MAGIC #         To be cumulatively summed as flat.
# MAGIC #     axis : int, default=None
# MAGIC #         Axis along which the cumulative sum is computed.
# MAGIC #         The default (None) is to compute the cumsum over the flattened array.
# MAGIC #     rtol : float, default=1e-05
# MAGIC #         Relative tolerance, see ``np.allclose``.
# MAGIC #     atol : float, default=1e-08
# MAGIC #         Absolute tolerance, see ``np.allclose``.
# MAGIC #     """
# MAGIC #     # borrow from scikit learn stable source...
# MAGIC #     #  https://github.com/scikit-learn/scikit-learn/blob/844b4be24d20fc42cc13b957374c718956a0db39/sklearn/utils/extmath.py#L1063
# MAGIC #     out = np.cumsum(arr, axis=axis)
# MAGIC #     expected = np.sum(arr, axis=axis)
# MAGIC #     #if not np.all(
# MAGIC #     #    np.isclose(
# MAGIC #     #        out.take(-1, axis=axis), expected, rtol=rtol, atol=atol, equal_nan=True
# MAGIC #     #    )
# MAGIC #     #):
# MAGIC #     #    warnings.warn(
# MAGIC #     #        "cumsum was found to be unstable: "
# MAGIC #     #        "its last element does not correspond to sum",
# MAGIC #     #        RuntimeWarning,
# MAGIC #     #    )
# MAGIC 
# MAGIC #     return out
# MAGIC 
# MAGIC # # fpr, fnr, _ = det_curve(df_predict['label'].ravel(), df_predict['score'].ravel())
# MAGIC # # print(fnr)
# MAGIC # # plot_DET_curve(fpr, fnr)
