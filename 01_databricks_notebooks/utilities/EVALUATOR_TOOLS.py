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
    ax[1].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax[1].set_xlim([0.0, 1.0])
    ax[1].set_ylim([0.0, 1.05])
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_title(f'Receiver operating characteristic (AUC {roc_auc:0.3f})')
    ax[1].legend(loc="lower right")
    ax[1].grid('both')

    return fig 


# COMMAND ----------

### EXPERIMENTAL / DISABLED CODE 
# We need some special plot functions but they aren't yet available in the sklearn version
# associated with this cluster / image.
# So, we consider replicating them in the code below.  At time of writing though, these 
# Functions are disabled because they have not received sufficient testing.

# import matplotlib.pyplot as plt
# import scipy as sp
# import numpy as np

  
# def plot_DET_curve(fpr, fnr, ax=None, *, title="DET Curve", pos_label=1, **kwargs):
#     """Plot visualization.
#     Parameters
#     ----------
#     ax : matplotlib axes, default=None
#         Axes object to plot on. If `None`, a new figure and axes is
#         created.
#     name : str, default=None
#         Name of DET curve for labeling. If `None`, use `estimator_name` if
#         it is not `None`, otherwise no labeling is shown.
#     **kwargs : dict
#         Additional keywords arguments passed to matplotlib `plot` function.
#     Returns
#     -------
#     display : :class:`~sklearn.metrics.plot.DetCurveDisplay`
#         Object that stores computed values.
#     """
#     # code borrowed from release version of sklearn...
#     #  https://github.com/scikit-learn/scikit-learn/blob/844b4be24/sklearn/metrics/_plot/det_curve.py#L190


#     if ax is None:
#         _, ax = plt.subplots()

#     (line_,) = ax.plot(
#         sp.stats.norm.ppf(fpr),
#         sp.stats.norm.ppf(fnr),
#     )
#     info_pos_label = (
#         f" (Positive label: {pos_label})" if pos_label is not None else ""
#     )

#     xlabel = "False Positive Rate" + info_pos_label
#     ylabel = "False Negative Rate" + info_pos_label
#     ax.set(xlabel=xlabel, ylabel=ylabel)

#     ax.legend(loc="lower right")

#     ticks = [0.001, 0.01, 0.05, 0.20, 0.5, 0.80, 0.95, 0.99, 0.999]
#     tick_locations = sp.stats.norm.ppf(ticks)
#     tick_labels = [
#         "{:.0%}".format(s) if (100 * s).is_integer() else "{:.1%}".format(s)
#         for s in ticks
#     ]
#     ax.set_xticks(tick_locations)
#     ax.set_xticklabels(tick_labels)
#     ax.set_xlim(-3, 3)
#     ax.set_yticks(tick_locations)
#     ax.set_yticklabels(tick_labels)
#     ax.set_ylim(-3, 3)
#     ax.grid('both')

#     ax_ = ax
#     figure_ = ax.figure
#     return figure_

# def det_curve(y_true, y_score, pos_label=None, sample_weight=None):
#     """Compute error rates for different probability thresholds.
#     .. note::
#        This metric is used for evaluation of ranking and error tradeoffs of
#        a binary classification task.
#     Read more in the :ref:`User Guide <det_curve>`.
#     .. versionadded:: 0.24
#     Parameters
#     ----------
#     y_true : ndarray of shape (n_samples,)
#         True binary labels. If labels are not either {-1, 1} or {0, 1}, then
#         pos_label should be explicitly given.
#     y_score : ndarray of shape of (n_samples,)
#         Target scores, can either be probability estimates of the positive
#         class, confidence values, or non-thresholded measure of decisions
#         (as returned by "decision_function" on some classifiers).
#     pos_label : int or str, default=None
#         The label of the positive class.
#         When ``pos_label=None``, if `y_true` is in {-1, 1} or {0, 1},
#         ``pos_label`` is set to 1, otherwise an error will be raised.
#     sample_weight : array-like of shape (n_samples,), default=None
#         Sample weights.
#     Returns
#     -------
#     fpr : ndarray of shape (n_thresholds,)
#         False positive rate (FPR) such that element i is the false positive
#         rate of predictions with score >= thresholds[i]. This is occasionally
#         referred to as false acceptance propability or fall-out.
#     fnr : ndarray of shape (n_thresholds,)
#         False negative rate (FNR) such that element i is the false negative
#         rate of predictions with score >= thresholds[i]. This is occasionally
#         referred to as false rejection or miss rate.
#     thresholds : ndarray of shape (n_thresholds,)
#         Decreasing score values.
#     See Also
#     --------
#     DetCurveDisplay.from_estimator : Plot DET curve given an estimator and
#         some data.
#     DetCurveDisplay.from_predictions : Plot DET curve given the true and
#         predicted labels.
#     DetCurveDisplay : DET curve visualization.
#     roc_curve : Compute Receiver operating characteristic (ROC) curve.
#     precision_recall_curve : Compute precision-recall curve.
#     Examples
#     --------
#     >>> import numpy as np
#     >>> from sklearn.metrics import det_curve
#     >>> y_true = np.array([0, 0, 1, 1])
#     >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
#     >>> fpr, fnr, thresholds = det_curve(y_true, y_scores)
#     >>> fpr
#     array([0.5, 0.5, 0. ])
#     >>> fnr
#     array([0. , 0.5, 0.5])
#     >>> thresholds
#     array([0.35, 0.4 , 0.8 ])
#     """
#     # link to recent scikit learn function...
#     #  https://github.com/scikit-learn/scikit-learn/blob/844b4be24/sklearn/metrics/_ranking.py#L237
#     fps, tps, thresholds = _binary_clf_curve(
#         y_true, y_score, pos_label=pos_label, sample_weight=sample_weight
#     )

#     if len(np.unique(y_true)) != 2:
#         raise ValueError(
#             "Only one class present in y_true. Detection error "
#             "tradeoff curve is not defined in that case."
#         )

#     fns = tps[-1] - tps
#     p_count = tps[-1]
#     n_count = fps[-1]

#     # start with false positives zero
#     first_ind = (
#         fps.searchsorted(fps[0], side="right") - 1
#         if fps.searchsorted(fps[0], side="right") > 0
#         else None
#     )
#     # stop with false negatives zero
#     last_ind = tps.searchsorted(tps[-1]) + 1
#     sl = slice(first_ind, last_ind)

#     # reverse the output such that list of false positives is decreasing
#     return (fps[sl][::-1] / n_count, fns[sl][::-1] / p_count, thresholds[sl][::-1])


# def _binary_clf_curve(y_true, y_score, pos_label=None, sample_weight=None):
#     """Calculate true and false positives per binary classification threshold.
#     Parameters
#     ----------
#     y_true : ndarray of shape (n_samples,)
#         True targets of binary classification.
#     y_score : ndarray of shape (n_samples,)
#         Estimated probabilities or output of a decision function.
#     pos_label : int or str, default=None
#         The label of the positive class.
#     sample_weight : array-like of shape (n_samples,), default=None
#         Sample weights.
#     Returns
#     -------
#     fps : ndarray of shape (n_thresholds,)
#         A count of false positives, at index i being the number of negative
#         samples assigned a score >= thresholds[i]. The total number of
#         negative samples is equal to fps[-1] (thus true negatives are given by
#         fps[-1] - fps).
#     tps : ndarray of shape (n_thresholds,)
#         An increasing count of true positives, at index i being the number
#         of positive samples assigned a score >= thresholds[i]. The total
#         number of positive samples is equal to tps[-1] (thus false negatives
#         are given by tps[-1] - tps).
#     thresholds : ndarray of shape (n_thresholds,)
#         Decreasing score values.
#     """
#     # link to recent scikit learn function...
#     #  https://github.com/scikit-learn/scikit-learn/blob/844b4be24/sklearn/metrics/_ranking.py#L237

#     # Filter out zero-weighted samples, as they should not impact the result
#     if sample_weight is not None:
#         sample_weight = _check_sample_weight(sample_weight, y_true)
#         nonzero_weight_mask = sample_weight != 0
#         y_true = y_true[nonzero_weight_mask]
#         y_score = y_score[nonzero_weight_mask]
#         sample_weight = sample_weight[nonzero_weight_mask]

#     # pos_label = _check_pos_label_consistency(pos_label, y_true)

#     # make y_true a boolean vector
#     y_true = y_true == pos_label

#     # sort scores and corresponding truth values
#     desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
#     y_score = y_score[desc_score_indices]
#     y_true = y_true[desc_score_indices]
#     if sample_weight is not None:
#         weight = sample_weight[desc_score_indices]
#     else:
#         weight = 1.0

#     # y_score typically has many tied values. Here we extract
#     # the indices associated with the distinct values. We also
#     # concatenate a value for the end of the curve.
#     distinct_value_indices = np.where(np.diff(y_score))[0]
#     threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

#     # accumulate the true positives with decreasing threshold
#     tps = stable_cumsum(y_true * weight)[threshold_idxs]
#     if sample_weight is not None:
#         # express fps as a cumsum to ensure fps is increasing even in
#         # the presence of floating point errors
#         fps = stable_cumsum((1 - y_true) * weight)[threshold_idxs]
#     else:
#         fps = 1 + threshold_idxs - tps
#     return fps, tps, y_score[threshold_idxs]

# def stable_cumsum(arr, axis=None, rtol=1e-05, atol=1e-08):
#     """Use high precision for cumsum and check that final value matches sum.
#     Parameters
#     ----------
#     arr : array-like
#         To be cumulatively summed as flat.
#     axis : int, default=None
#         Axis along which the cumulative sum is computed.
#         The default (None) is to compute the cumsum over the flattened array.
#     rtol : float, default=1e-05
#         Relative tolerance, see ``np.allclose``.
#     atol : float, default=1e-08
#         Absolute tolerance, see ``np.allclose``.
#     """
#     # borrow from scikit learn stable source...
#     #  https://github.com/scikit-learn/scikit-learn/blob/844b4be24d20fc42cc13b957374c718956a0db39/sklearn/utils/extmath.py#L1063
#     out = np.cumsum(arr, axis=axis)
#     expected = np.sum(arr, axis=axis)
#     #if not np.all(
#     #    np.isclose(
#     #        out.take(-1, axis=axis), expected, rtol=rtol, atol=atol, equal_nan=True
#     #    )
#     #):
#     #    warnings.warn(
#     #        "cumsum was found to be unstable: "
#     #        "its last element does not correspond to sum",
#     #        RuntimeWarning,
#     #    )

#     return out

# # fpr, fnr, _ = det_curve(df_predict['label'].ravel(), df_predict['score'].ravel())
# # print(fnr)
# # plot_DET_curve(fpr, fnr)


