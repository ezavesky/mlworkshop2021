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

# expects either a list, DenseArray or SparseArray type
udf_last_prediction = F.udf(lambda v: v[-1] if type(v)==list else v.toArray().tolist()[-1], T.DoubleType())

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

import os
import sys
import pickle
import numpy as np
import pandas as pd
import importlib
from matplotlib.ticker import PercentFormatter # comment out for km4
from matplotlib import pyplot as plt

# 45-degree straight line (baseline) for gains chart
def abline(slope, intercept, ax): #, fig, ax): # = plt.gca()):
    """Plot a line from slope and intercept"""
    x_vals = np.array(ax.get_xlim())
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, '--') # when ax is specified (and not just gca)

# one single area's models
def lifts_metrics_varimps_for_single_area_multi_depths(pickle_path, dict_run_tags, split_tag, target_area, model_tag):
    '''
    dict_run_tags = dict of run_tags specifying depth, specifically run_tag02 through run_tag10, e.g., dict_run_tags['02'] = 'using_500_mtl_garage_area09southtx_depth02'
    target_area = list of area; single area, but has to be in a list
    '''
    df_lifts = pd.DataFrame()
    df_metrics = pd.DataFrame()
    df_varimps = pd.DataFrame()

    all_depths = dict_run_tags.keys()
    for depth in all_depths:
        df_lift, df_metric, df_varimp, form_type_sorted, test_type_sorted, capture_type_sorted, prod_type_sorted = get_perfs(pickle_path, split_tag, target_area, dict_run_tags[depth], model_tag)
        add_run_tag(df_lift, depth)
        add_run_tag(df_metric, depth)
        add_run_tag(df_varimp, depth)
        df_lifts = pd.concat([df_lifts, df_lift])
        df_metrics = pd.concat([df_metrics, df_metric])
        df_varimps = pd.concat([df_varimps, df_varimp])
    return (df_lifts, df_metrics, df_varimps)

# get gains for a target area from all 12 models
def get_area_stats_target_areas_run_tags(dict_area_pickles, target_areas, run_tags):
    dict_area_lifts = dict()
    dict_area_metrics = dict()
    dict_area_varimps = dict()
    for target_area in target_areas:
        df_lifts = pd.DataFrame()
        df_metrics = pd.DataFrame()
        df_varimps = pd.DataFrame()
        print('For area: ' + target_area)
        for run_tag in run_tags:
            df_lifts = pd.concat([df_lifts, dict_area_pickles[target_area][run_tag]['df_lift']])
            df_metrics = pd.concat([df_metrics, dict_area_pickles[target_area][run_tag]['df_metric']])
            df_varimps = pd.concat([df_varimps, dict_area_pickles[target_area][run_tag]['df_varimp']])
        dict_area_lifts[target_area] = df_lifts
        dict_area_metrics[target_area] = pd.concat([df_metrics, dict_area_pickles[target_area][run_tag]['df_metric']])
        dict_area_varimps[target_area] = pd.concat([df_varimps, dict_area_pickles[target_area][run_tag]['df_varimp']])
    return(dict_area_lifts, dict_area_metrics, dict_area_varimps)

# read pickles for each of run_tag and target_area
def read_pickles_target_areas_run_tags(pickle_path, split_tag, target_areas, run_tags, model_tag):
    dict_area_pickles = dict()
    for target_area in target_areas:
        dict_area_pickles[target_area] = dict() # each target area has its own dict for 12 run tags
        target_area_list = [target_area] * len(run_tags)
        for run_tag in run_tags:
            dict_area_pickles[target_area][run_tag] = dict()
            # order of get_perfs() return value
            # df_lift, df_metric, df_varimp, form_type_sorted, test_type_sorted, capture_type_sorted, prod_type_sorted = get_perfs(pickle_path, split_tag, target_areas, run_tag, model_tag)
            df_lift, df_metric, df_varimp, form_type_sorted, test_type_sorted, capture_type_sorted, prod_type_sorted = get_perfs(pickle_path, split_tag, target_area_list, run_tag, model_tag)
            depth_subset = str.split(run_tag, '_regional2_')[-1]
            add_run_tag(df_lift, depth_subset)
            add_run_tag(df_metric, depth_subset)
            add_run_tag(df_varimp, depth_subset)
            dict_area_pickles[target_area][run_tag]['df_lift'] = df_lift
            dict_area_pickles[target_area][run_tag]['df_metric'] = df_metric
            dict_area_pickles[target_area][run_tag]['df_varimp'] = df_varimp
    return(dict_area_pickles)

# for each market, see its performance and overfit
# aggregate of existing functions
def show_area_performance(pickle_path, split_tag, target_areas, run_tag, model_tag):
    # read in pickles (mainly for df_lift)
    print("Read in pickle")
    df_lift, df_metric, df_varimp, form_type_sorted, test_type_sorted, capture_type_sorted, prod_type_sorted = get_perfs(pickle_path, split_tag, target_areas, run_tag, model_tag)

    # calc perf eval
    print("Show gains")
    show_firstN(df_lift = df_lift, test_type_sorted = test_type_sorted, capture_type_sorted = capture_type_sorted, prod_type_sorted = prod_type_sorted, grp_first = 0.1, grp_last = 0.4)
    
    # compare sales counts
    print("Compare sales counts")
    dict_sales = compare_sales_counts(df_lift, prod_type_sorted, test_type_sorted)
    
    # plot compare (multiple pages)
    plot_sales_counts(dict_sales, prod_type_sorted, test_type_sorted, target_prod = 'mob', target_test = 'train', indiv = True)
    plot_sales_counts(dict_sales, prod_type_sorted, test_type_sorted, target_prod = 'mob', target_test = 'test1', indiv = True)
    plot_sales_counts(dict_sales, prod_type_sorted, test_type_sorted, target_prod = 'mob', target_test = 'test2', indiv = True)
    
    # show overfit
    print("Show overfit")
    show_overfits(df_lift = df_lift, capture_type_sorted = capture_type_sorted, prod_type_sorted = prod_type_sorted)
    compare_diff_gains(df_lift = df_lift, capture_type_sorted = capture_type_sorted, prod_type_sorted = prod_type_sorted, index = 'group_tag', which_test = 'test1', compare_models_in_multi = False, overfit_thresh  = -0.02)
    compare_diff_gains(df_lift = df_lift, capture_type_sorted = capture_type_sorted, prod_type_sorted = prod_type_sorted, index = 'group_tag', which_test = 'test2', compare_models_in_multi = False, overfit_thresh  = -0.02)
    


# national vs. market groups aggregation comparison
# check for (1) overall number of visits (should match), (2) sales counts
def compare_sales_counts(df_lift, prod_type_sorted, test_type_sorted):
    dict3_comp = dict() # dict of dict of dict of df (key: prod type)
    for prod_type in prod_type_sorted:
        dict2_comp = dict() # dict of dict of df (key: test type)
        for test_type in test_type_sorted:
            dict_comp = dict() # dict of df (key: decile group)
            for decile in range(10):
                decile_group = (decile + 1)/10.0
                df_lift_decile = df_lift.loc[(df_lift.group == decile_group) & (df_lift.prod_type == prod_type) & (df_lift.test_type == test_type), ['tag', 'test_type', 'prod_type', 'size_grp', 'resp_num_grp', 'resp_rate_grp', 'gain_grp', 'lift_grp', 'size_cum', 'resp_num_cum', 'resp_rate_cum', 'gain_cum', 'lift_cum']].sort_values(by = ['test_type', 'prod_type', 'tag'])
                df_lift_decile_indiv = df_lift_decile.loc[(df_lift_decile.tag != 'national'), ['size_cum', 'resp_num_cum']].cumsum().tail(1)
                df_lift_decile_indiv = df_lift_decile_indiv.rename(columns = {"size_cum": "size_cum_indiv", "resp_num_cum": "resp_num_cum_indiv"})
                df_lift_decile_natio = df_lift_decile.loc[(df_lift_decile.tag == 'national'), ['size_cum', 'resp_num_cum']].cumsum().tail(1) 
                df_lift_decile_natio = df_lift_decile_natio.rename(columns = {"size_cum": "size_cum_natio", "resp_num_cum": "resp_num_cum_natio"})
                df_lift_combined = pd.concat([df_lift_decile_indiv, df_lift_decile_natio], axis = 1)
                df_lift_combined['group'] = decile_group
                dict_comp[decile] = df_lift_combined
            df_comp = pd.concat(v for v in dict_comp.values())
            dict2_comp[test_type] = df_comp
        dict3_comp[prod_type] = dict2_comp
    return dict3_comp

# plot sales counts for national vs. market groups (individual) for each prod_type and test_type individually!
def plot_sales_counts(dict3_comp, prod_type_sorted, test_type_sorted, target_prod = 'mob', target_test = 'train', indiv = False):
    if (indiv):
        for n, prod_type in enumerate(prod_type_sorted):
            dict_dict_comp = dict3_comp[prod_type]
            for k, test_type in enumerate(test_type_sorted):
                df = dict_dict_comp[test_type]
                if (prod_type == target_prod) & (test_type == target_test):
                    df.loc[:, ['group', 'resp_num_cum_indiv', 'resp_num_cum_natio']].plot(kind = 'bar', x = 'group', title = 'Sales counts comparison for ' + prod_type + '/' + test_type)
        plt.show()
    else:
        fig, axes = plt.subplots(nrows = len(prod_type_sorted), ncols = len(test_type_sorted), figsize = (20, 20))
        for n, prod_type in enumerate(prod_type_sorted):
            dict_dict_comp = dict3_comp[prod_type]
            for k, test_type in enumerate(test_type_sorted):
                df = dict_dict_comp[test_type]
                ax = axes[n, k]
                df.loc[:, ['group', 'resp_num_cum_indiv', 'resp_num_cum_natio']].plot(kind = 'bar', x = 'group', ax = ax, title = prod_type + '/' + test_type)
                ax.xaxis.set_label_text('')
                fig.suptitle('Sales counts comparison', fontsize = 16)
        plt.show()


# plot gains chart
def plot_gains(df, wcols, val, ax, title):
    ''' df: pandas df
        wcols: cols to be "widened" in pivot after which they are plotting vals
        val: actual values to fill in wcols (after pivoting)
        fig, ax: canvas and subplot's ax
    '''
    df_wide = df.pivot(index = 'group', columns = wcols, values = val)
    df_wide.plot(kind = 'line', title = title, ax = ax, grid = 'on', rasterized = True)
    abline(1, 0, ax) #, fig = fig, ax = ax)

# plot gains for first N groups (in bar)
def plot_firstN(df, grp_first, grp_last, wcols, val, ax, title):
    #df_firstN = df[(df.group == grp_n1) | (df.group == grp_n2)]
    df_firstN = df[(df.group >= grp_first) & (df.group <= grp_last)]
    df_wide = df_firstN.pivot(index = 'group', columns = wcols, values = val)
    df_wide.plot(kind = 'bar', title = title, ax = ax, grid = 'on')

# plot variable importance
def plot_varimp(df_vimp, topn, ax, title):
    df_vimp.head(topn).plot(kind = 'barh', x = 'var', y = 'rel', ax = ax, title = title)
    ax.invert_yaxis()
    ax.xaxis.set_tick_params(labelsize = 2)

# plot model metrics for first N groups (in bar)
def plot_metric(df, x, y, ax, title):
    df.plot(kind = 'bar', x = x, y = y, ax = ax, title = title) 
    ax.yaxis.set_major_formatter(PercentFormatter(xmax = 1.0))

# read in model artifacts object pickle file
def read_model_obj(pickle_path, split_tag, target_area, run_tag, model_tag):
    #with open('/opt/data/share05/sandbox/sandbox142/sndbx_scripts/jl939a/ihx/phase4.0.1/logs/' + run_tag + '/model_objects_1_' + model_tag + '.pickle', 'rb') as file:
    #with open(pickle_path + '/' + run_tag + '/model_objects_1_' + model_tag + '.pickle', 'rb') as file:
    with open(pickle_path + '/' + run_tag + '/model_objects_1_' + split_tag + '_' + target_area + '_' + run_tag + '_' + model_tag + '.pickle', 'rb') as file:
        vars_to_drop, varimp_dict, lift_dict, metric_dict = pickle.load(file)

        # remove suffix from var column in varimp_df
        #varimp_df['var'] = varimp_df['var'].str.replace('_imputed', '')
        #varimp_df['var'] = varimp_df['var'].str.replace('_index', '')
    return [vars_to_drop, varimp_dict, lift_dict, metric_dict]

def get_perfs(pickle_path, split_tag, target_areas, run_tag, model_tag):
    varimp_dict_all = {}
    lift_dict_all = {}
    metric_dict_all = {}
    
    for area in target_areas:
    # read model object pickle files
        release = area
        acv_form = area
        vars_to_drop, varimp_dict, lift_dict, metric_dict = read_model_obj(pickle_path, split_tag, area, run_tag, model_tag)
    
        for top_n in lift_dict.keys():
            lift_df = lift_dict[top_n]
            lift_df['tag'] = area
            lift_df['release'] = release
            lift_df['form_type'] = acv_form
            lift_df['test_type'] = top_n.split('_')[0]
            #lift_df['subset_type'] = top_n.split('_')[1]
            lift_df['capture_type'] = top_n.split('_')[2]
            lift_df['prod_type'] = np.where(len(top_n.split('_')) > 3, top_n.split('_')[-1], 'all')
            lift_dict_all[top_n + '_' + release + '_' + acv_form] = lift_df
    
        for top_n in metric_dict.keys():
            metric_df = pd.DataFrame(metric_dict.items(), columns = ['test_type', 'metric'])
            metric_df['tag'] = area
            metric_df['release'] = release
            metric_df['form_type'] = acv_form
            metric_dict_all[release + '_' + acv_form] = metric_df
    
        for top_n in varimp_dict.keys():
            varimp_df = varimp_dict[top_n]
            varimp_df['tag'] = area 
            varimp_df['release'] = release
            varimp_df['form_type'] = acv_form
            # remove suffix from var column in varimp_df
            varimp_df['var'] = varimp_df['var'].str.replace('_imputed', '')
            varimp_df['var'] = varimp_df['var'].str.replace('_index', '')
            varimp_dict_all[top_n + '_' + release + '_' + acv_form] = varimp_df
    
    # make a long dataframe
    df_lift = pd.DataFrame()
    df_metric = pd.DataFrame()
    df_varimp = pd.DataFrame()
    
    for lift_tag in lift_dict_all.keys():
        df_lift = df_lift.append(lift_dict_all[lift_tag])
    
    for metric_tag in metric_dict_all.keys():
        df_metric = df_metric.append(metric_dict_all[metric_tag])
    
    for varimp_tag in varimp_dict_all.keys():
        df_varimp = df_varimp.append(varimp_dict_all[varimp_tag])
    
    # sort unique values (in place by default)
    form_type_sorted = sorted(df_lift.form_type.unique())
    test_type_sorted = sorted(df_lift.test_type.unique())
    capture_type_sorted = sorted(df_lift.capture_type.unique())
    prod_type_sorted = sorted(df_lift.prod_type.unique())

    return [df_lift, df_metric, df_varimp, form_type_sorted, test_type_sorted, capture_type_sorted, prod_type_sorted]

def show_gains(df_lift, test_type_sorted, capture_type_sorted, prod_type_sorted):
    # NOTE: for some reason, assigning defaults and using them w/o specifying values, hence (attempting to use defaults) in function call, e.g., show_gains() didn't work as intended...
    # plot gain charts for a given model_tag for all capture types (sales/values) and prod types (all/mob/bb), hence 2 x 3 = 6
    # each plot showing lines per each test types (3)
    for test_type in test_type_sorted:
        df_test_type = df_lift[df_lift.test_type == test_type]
        fig, axes = plt.subplots(nrows = len(capture_type_sorted), ncols = len(prod_type_sorted), figsize = (20, 20))
        for k, capture_type in enumerate(capture_type_sorted):
            for n, prod_type in enumerate(prod_type_sorted):
                df = df_test_type[(df_test_type.capture_type == capture_type) & (df_test_type.prod_type == prod_type)]  # have to assigne to a new df for plotting (not sure why)
                #plot_gains(df, wcols = 'tag', val = 'gain_cum', ax = axes[k, n], title = capture_type + '/' + prod_type)
                plot_gains(df, wcols = 'tag', val = 'gain_cum', ax = axes[n], title = capture_type + '/' + prod_type)
        fig.suptitle('Cumulative Gains for ' + test_type, fontsize=16)
        plt.show()

def show_firstN(df_lift, test_type_sorted, capture_type_sorted, prod_type_sorted, grp_first = 0.1, grp_last = 0.2):
    # NOTE: for some reason, assigning defaults and using them w/o specifying values, hence (attempting to use defaults) in function call, e.g., show_gains() didn't work as intended...
    for test_type in test_type_sorted:
        df_test_type = df_lift[df_lift.test_type == test_type]
        fig, axes = plt.subplots(nrows = len(capture_type_sorted), ncols = len(prod_type_sorted), figsize = (20, 20))
        for k, capture_type in enumerate(capture_type_sorted):
            for n, prod_type in enumerate(prod_type_sorted):
                df = df_test_type[(df_test_type.capture_type == capture_type) & (df_test_type.prod_type == prod_type)]
                #plot_firstN(df, grp_first = grp_first, grp_last = grp_last, wcols = 'tag', val = 'gain_cum', ax = axes[k, n], title = capture_type + '/' + prod_type)
                plot_firstN(df, grp_first = grp_first, grp_last = grp_last, wcols = 'tag', val = 'gain_cum', ax = axes[n], title = capture_type + '/' + prod_type)
        fig.suptitle('Top 2 Decile Gains for ' + test_type, fontsize=16)
        plt.show()

## plot variable importance
def show_varimp(tags, df_varimp, topn = 30):
    fig, axes = plt.subplots(nrows = len(tags), ncols = 1, figsize = (20, 20))
    for k, tag in enumerate(tags):
        df = df_varimp[df_varimp.tag == tag]
        if (len(tags) == 1):
            plot_varimp(df, topn = topn, ax = axes, title = tag)
        else:
            plot_varimp(df, topn = topn, ax = axes[k], title = tag)
    fig.suptitle('Variable Importance: top ' + str(topn) + ' vars') 
    plt.show()

## plot default model metrics (rmse/auc)
def show_metrics(df_metric, test_type_sorted):
    # each subplot showing metrics for each test type
    fig, axes = plt.subplots(nrows = 1, ncols = len(test_type_sorted), figsize = (20, 10))
    for k, test_type in enumerate(test_type_sorted): 
        df = df_metric[df_metric.test_type == test_type]
        df = df.sort_values(['tag', 'test_type'])
        if (len(test_type_sorted) == 1):
            plot_metric(df, x = 'tag',  y = 'metric', ax = axes, title = test_type)
        else:
            plot_metric(df, x = 'tag',  y = 'metric', ax = axes[k], title = test_type)
    plt.show()

## plot model metrics for comparison between differen data sets (train/test1/test2)
def show_metrics2(df_metric):
    mod_tags = df_metric.tag.unique() 
    fig, axes = plt.subplots(nrows = 1, ncols = len(mod_tags), figsize = (20, 10))
    for k, tag in enumerate(mod_tags):
        df = df_metric[df_metric.tag == tag]
        df = df.sort_values(['tag', 'test_type'])
        if (len(mod_tags) == 1):
            plot_metric(df, x = 'test_type', y = 'metric', ax = axes, title = tag)
        else:
            plot_metric(df, x = 'test_type', y = 'metric', ax = axes[k], title = tag)
    plt.show()

## plot metric difference between train and each test set (hence 1 plot for each model tag, overall 1 row x 2 columns)
def compare_metrics(df_metric, test_type_sorted):
    mod_tags = df_metric.tag.unique()
    df_wide = df_metric.pivot(index = 'tag', columns = 'test_type', values = 'metric') 
    df_wide['train_test1'] = (df_wide['test1'] - df_wide['train'])/(df_wide['train'])
    df_wide['train_test2'] = (df_wide['test2'] - df_wide['train'])/(df_wide['train'])
    df_wide = df_wide.reset_index() 
    fig, axes = plt.subplots(nrows = 1, ncols = len(test_type_sorted) - 1, figsize = (20, 10))
    for n, test_type in enumerate(test_type_sorted):
        if (test_type != 'train'):
            plot_metric(df_wide, x = 'tag', y = 'train_test' + str(n + 1), ax = axes[n], title = test_type)
    fig.suptitle('Model Metric Difference' , fontsize=16)
    plt.show()

## quick fix to add run_tag info into tag column in df_lift, df_varimp, etc from different settings
## e.g., compare df_lifts from different folders (w/ different run_tag values)
def add_run_tag(df, run_tag):
    df['tag'] = df['tag'] + '_' + run_tag
    #df['tag'].replace(['model'], 'model' + run_tag, regex=True, inplace=True)

## plot train/test1/test2 gains in one plot
def show_overfits(df_lift, capture_type_sorted, prod_type_sorted):
    mod_tags = df_lift.tag.unique()
    for tag in mod_tags: 
        df_tag_type = df_lift[df_lift.tag == tag]
        fig, axes = plt.subplots(nrows = len(capture_type_sorted), ncols = len(prod_type_sorted), figsize = (20, 20))
        for k, capture_type in enumerate(capture_type_sorted):
            for n, prod_type in enumerate(prod_type_sorted):
                df = df_tag_type[(df_tag_type.capture_type == capture_type) & (df_tag_type.prod_type == prod_type)]
                plot_firstN(df, grp_first = 0.1, grp_last = 0.4, wcols = 'test_type', val = 'gain_cum', ax = axes[n], title = capture_type + '/' + prod_type)
        fig.suptitle('Top N Decile Gains for ' + tag, fontsize=16)
        plt.show()

# plot gains for first N groups (in line)
def plot_diff_gains(df, grp_first, grp_last, index, wcols, val, ax, title, which_test, compare_models_in_multi = True, overfit_thresh = -0.02):
    df_firstN = df[(df.group >= grp_first) & (df.group <= grp_last)]
    df_wide = df_firstN.pivot(index = index, columns = wcols, values = val)
    df_wide['train_test1'] = (df_wide['test1'] - df_wide['train'])/(df_wide['train']) 
    df_wide['train_test2'] = (df_wide['test2'] - df_wide['train'])/(df_wide['train']) 
    if compare_models_in_multi:
        if which_test == 'both':
            df_wide[['train_test1', 'train_test2']].plot(kind = 'line', title = title, ax = ax, grid = 'on').axhline(y = overfit_thresh, color = 'black')
            ax.yaxis.set_major_formatter(PercentFormatter(xmax = 1.0))
            #ax.hlines(y = overfit_thresh, xmin = grp_first, xmax = grp_last, color = 'r', linestyle = '-')
        elif which_test == 'test1':
            df_wide['train_test1'].plot(kind = 'line', title = title, ax = ax, grid = 'on').axhline(y = overfit_thresh, color = 'black')
            ax.yaxis.set_major_formatter(PercentFormatter(xmax = 1.0))
            #ax.hlines(y = overfit_thresh, xmin = grp_first, xmax = grp_last, color = 'r', linestyle = '-')
        elif which_test == 'test2':
            df_wide['train_test2'].plot(kind = 'line', title = title, ax = ax, grid = 'on').axhline(y = overfit_thresh, color = 'black') 
            ax.yaxis.set_major_formatter(PercentFormatter(xmax = 1.0))
            #ax.hlines(y = overfit_thresh, xmin = grp_first, xmax = grp_last, color = 'r', linestyle = '-')
    else:
        df = df_wide.reset_index()
        df['group'], df['tag'] = df['group_tag'].str.split('_', 1).str
        df = df.drop(labels = ['group_tag'], axis = 1) # for use in km4 (older pandas version)
        #df = df.drop(columns = ['group_tag'])
        # reminder that Python wants wide form (untidy) data for plotting grouped bar/line chart
        if which_test == 'test1':
            df_wide = df.pivot(index = 'group', columns = 'tag', values = 'train_test1')
        elif which_test == 'test2': 
            df_wide = df.pivot(index = 'group', columns = 'tag', values = 'train_test2')
        df_wide.plot(kind = 'line', title = title, ax = ax, grid = 'on').axhline(y = overfit_thresh, color = 'black')  
        ax.yaxis.set_major_formatter(PercentFormatter(xmax = 1.0))
        #ax.hlines(y = overfit_thresh, xmin = grp_first, xmax = grp_last, color = 'r', linestyle = '-')

## plot diff in gains (to show magnitude of potential overfit) in multi plots (e.g., one plot per one model tag)
def show_diff_gains(df_lift, capture_type_sorted, prod_type_sorted, index, which_test, overfit_thresh = -0.02):
    mod_tags = df_lift.tag.unique()
    for tag in mod_tags: 
        df_tag_type = df_lift[df_lift.tag == tag]
        fig, axes = plt.subplots(nrows = len(capture_type_sorted), ncols = len(prod_type_sorted), figsize = (20, 20))
        for k, capture_type in enumerate(capture_type_sorted):
            for n, prod_type in enumerate(prod_type_sorted):
                df = df_tag_type[(df_tag_type.capture_type == capture_type) & (df_tag_type.prod_type == prod_type)]
                plot_diff_gains(df, grp_first = 0.1, grp_last = 1.0, index = index, wcols = 'test_type', val = 'gain_cum', ax = axes[n], title = capture_type + '/' + prod_type, which_test = which_test, overfit_thresh = overfit_thresh)
        fig.suptitle('Top N Decile Difference in Gains for ' + tag, fontsize=16)
        plt.show()

## plot diff in gains (to show magnitude of potential overfit) in one plot
def compare_diff_gains(df_lift, capture_type_sorted, prod_type_sorted, index, which_test = 'test1', compare_models_in_multi = False, overfit_thresh = -0.02):
    mod_tags = df_lift.tag.unique()
    df_lift['group_tag'] = df_lift['group'].map(str) + '_' + df_lift['tag'] 
    fig, axes = plt.subplots(nrows = len(capture_type_sorted), ncols = len(prod_type_sorted), figsize = (20, 20))
    for k, capture_type in enumerate(capture_type_sorted):
        for n, prod_type in enumerate(prod_type_sorted):
            df = df_lift[(df_lift.capture_type == capture_type) & (df_lift.prod_type == prod_type)]
            plot_diff_gains(df, grp_first = 0.1, grp_last = 1.0, index = index, wcols = 'test_type', val = 'gain_cum', ax = axes[n], title = capture_type + '/' + prod_type, which_test = which_test, compare_models_in_multi = compare_models_in_multi, overfit_thresh = overfit_thresh)
    fig.suptitle('Top N Decile Difference in Gains' , fontsize=16)
    plt.show()

# calc train-test1 and train-test2 for target values, e.g., metrics from df_metrics or gains from df_lift
def calc_test_minus_train(df, wcols, val, capture_type = 'sales', prod_type = 'mob'):
    df = df[(df.capture_type == capture_type) & (df.prod_type == prod_type)]
    df['group_tag'] = df['group'].map(str) + '_' + df['tag'] 
    df_wide = df.pivot(index = 'group_tag', columns = wcols, values = val)
    df_wide['train_test1'] = (df_wide['test1'] - df_wide['train'])/(df_wide['train']) 
    df_wide['train_test2'] = (df_wide['test2'] - df_wide['train'])/(df_wide['train']) 
    df_wide = df_wide.reset_index()
    df_wide['group'], df_wide['tag'] = df_wide['group_tag'].str.split('_', 1).str
    df_wide = df_wide.drop(columns = ['group_tag'])
    return(df_wide)

# show train-test results at select decile and order by its size
def show_test_minus_train(df_wide, target_group, by_which_test, overfit_thresh):
    df_wide = df_wide[df_wide.group == str(target_group)]
    df_wide = df_wide.astype({by_which_test: 'float64'}).sort_values(by = by_which_test, ascending = False)
    if (overfit_thresh is not None): 
        df_wide = df_wide[df_wide[by_which_test] > overfit_thresh]
    return(df_wide)


# COMMAND ----------


