# Databricks notebook source
import pandas as pd
import numpy as np

# this is a custom evaluator for cumulative gain scoring

def cumulative_gain_score(actual = 'final_response', predicted = 'score', data_pd=None):
    '''
        :param actual: Actual target values passed as a python list.
        :param predicted: List of probabilities of positive class.
        :return: Return the score value as float type.
    '''
    if data_pd is None:
        # intialise data of lists.
        data = {'final_response': actual, 'score':predicted}
        # Create DataFrame
        data_pd = pd.DataFrame(data)
        
    data_ordered = data_pd.sort_values(by = ['score'], ascending = False)
    data_ordered['row_num'] = range(len(data_ordered))
    nrows = len(data_ordered)
    resp_tot = data_ordered.loc[:, 'final_response'].sum()
    col_names = ['dummy', 'group', 'size_grp', 'resp_num_grp', 'resp_rate_grp', 'gain_grp', 'lift_grp', 'size_cum', 'resp_num_cum', 'resp_rate_cum', 'gain_cum', 'lift_cum']
    y = pd.DataFrame(columns = col_names)

    #Initializing group array
    at = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    y['group'] = at
    y['dummy'] = 'lift'
    # for each at, calculate group level lift breakdown, whose cumulative sum should match results from above
    for i in range(len(at) - 1):
        x_grp = data_ordered.loc[(data_ordered.row_num >= at[i] * nrows) & (data_ordered.row_num < at[i + 1] * nrows), :]
        size = len(x_grp)
        resp_num = x_grp.loc[:, 'final_response'].sum() 
        resp_rate = resp_num/(size * 1.0)
        lift = (resp_num/(resp_tot * 1.0)) * (nrows/(size * 1.0))
        y.loc[y.group == at[i], 'size_grp'] = size
        y.loc[y.group == at[i], 'resp_num_grp'] = resp_num
        y.loc[y.group == at[i], 'resp_rate_grp'] = round(resp_rate, 4)
        y.loc[y.group == at[i], 'gain_grp'] = round(1.0 * resp_num/resp_tot, 4) #* n_grp 
        y.loc[y.group == at[i], 'lift_grp'] = round(lift, 4)

        x_cum = data_ordered.loc[(data_ordered.row_num < at[i + 1] * nrows), :]
        size = len(x_cum)
        resp_num = x_cum.loc[:, 'final_response'].sum() 
        resp_rate = resp_num/(size * 1.0)
        lift = (resp_num/(resp_tot * 1.0)) * (nrows/(size * 1.0))
        y.loc[y.group == at[i], 'size_cum'] = size
        y.loc[y.group == at[i], 'resp_num_cum'] = resp_num
        y.loc[y.group == at[i], 'resp_rate_cum'] = round(resp_rate, 4)
        y.loc[y.group == at[i], 'gain_cum'] = round(1.0 * resp_num/resp_tot, 4) #* n_grp 
        y.loc[y.group == at[i], 'lift_cum'] = round(lift, 4)
   
    y = y[y.group != 1]
    y['group'] = at[1:]

    #extracting cumulative gain for top 2 decile
    cum_gain = y.iloc[7]['gain_cum']
    return cum_gain

# COMMAND ----------

from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasLabelCol, HasRawPredictionCol
from pyspark.ml.linalg import DenseVector

class EvaluatorCustomBase(HasLabelCol, HasRawPredictionCol):
    # https://spark.apache.org/docs/latest/api/python/_modules/pyspark/ml/evaluation.html#BinaryClassificationEvaluator
    # refactored 08/28/21
    rawPredictionCol = Param(Params._dummy(), "rawPredictionCol",
                       "raw prediction column ",
                       typeConverter=TypeConverters.toString)
    labelCol = Param(Params._dummy(), "labelCol",
                       "label column (truth) ",
                       typeConverter=TypeConverters.toString)
    metricName = Param(Params._dummy(), "metricName",
                       "metric name in evaluation ",
                       typeConverter=TypeConverters.toString)

    def __init__(self, rawPredictionCol="rawPrediction", labelCol="label",
                 metricName="areaUnderROC"):
        super(EvaluatorCustomBase, self).__init__()
        self.setParams(rawPredictionCol, labelCol, metricName)

    def setMetricName(self, value):
        return self._set(metricName=value)

    def getMetricName(self):
        return self.getOrDefault(self.metricName)

    def setLabelCol(self, value):
        return self._set(labelCol=value)

    def getLabelCol(self):
        return self.getOrDefault(self.labelCol)

    def setRawPredictionCol(self, value):
        return self._set(rawPredictionCol=value)

    def getRawPredictionCol(self):
        return self.getOrDefault(self.rawPredictionCol)

    def setParams(self, rawPredictionCol="rawPrediction", labelCol="label",
                  metricName="unknown"):
        return self._set(rawPredictionCol=rawPredictionCol, metricName=metricName, labelCol=labelCol)

    def evaluate(self, dataset, params=None):
        if params is None:
            params = dict()
        if isinstance(params, dict):
            if params:
                return self.copy(params)._evaluate(dataset)
            else:
                return self._evaluate(dataset)
        else:
            raise ValueError("Params must be a param map but got %s." % type(params))

    def isLargerBetter(self):
        return True

    def _flatten_to_koalas(self, dataset):
        dim_test = dataset.limit(1).select([self.getRawPredictionCol()]).collect()
        kdf_test = ks.DataFrame(dataset.limit(1))[self.getRawPredictionCol()][0]
        # print(f"FOUND TYPE: {type(kdf_test)}, size: {dim_test}, values: {kdf_test}")
        col_pred = self.getRawPredictionCol()
        col_label = self.getLabelCol()
        # needs to be flattened to normal vector
        try:
            kdf_test = kdf_test.values
            # if type(kdf2[self.getRawPredictionCol()][0]) == pyspark.ml.linalg.DenseVector:
            return (dataset.rdd.map(lambda x:[float(x[col_pred][-1]), float(x[col_label])])
                        .toDF([col_pred, col_label]))

        except AttributeError as e:
            print(f"EXCEPTION: {kdf_test}")
            pass
        return dataset.select([col_pred, col_label])

    def _evaluate(self, dataset):
        # y_label, y_pred = self._evaluate_prep(dataset)
        raise NotImplementedError()

udf_last_prediction = F.udf(lambda v: v.toArray().tolist()[-1], T.DoubleType())

class CuluativeGainEvaluator(EvaluatorCustomBase):
    def isLargerBetter(self):
        return True

    def _evaluate(self, dataset):
        col_pred = self.getRawPredictionCol()
        col_label = self.getLabelCol()

        # expecting a dense vector for probabilities
        # expecting a single int/float for label (0/1)
        col_score = 'HIDDEN_score'
        dataset = dataset.withColumn(col_score, udf_last_prediction(F.col(col_pred)))
        
        # run through local function with minimal pandas dataframe
        return cumulative_gain_score(
            data_pd=dataset.select(F.col(col_label).alias('final_response'), 
                                   F.col(col_score).alias('score')).toPandas())


# COMMAND ----------

if False:
    # Example code that runs spark-wrapped version of the custom metric...
    list_score = np.random.random_sample(size = 1000)
    cut_line = int(len(list_score)*0.25)
    data_rand = [(DenseVector([1-list_score[i], list_score[i]]), 
                  1 if i < cut_line else 0) 
                 for i in range(len(list_score))]
    sdf = spark.createDataFrame(data_rand, ['pred', 'label'])
    evalcgain = CuluativeGainEvaluator(rawPredictionCol="pred", labelCol='label')
    score = evalcgain.evaluate(sdf)
    print(f"Cumulative Gain in Quartile: {score}")

# COMMAND ----------


