#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/25 07:45
# @Author  : Gongle
# @File    : bin_ml_metric.py
# @Version : 1.0
# @Desc    : None


ML_model_utils - -Local(plotting)

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold, cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from datetime import datetime
import seaborn as sns
import re
import os

plt.rcParams['font.sans-serif'] = ['SimHei']


def ivAutoCombing(DataFrame, variable, target, valueContinuous=True, equalWidth=False, bins=10,
                  replace=False, minorityReplace=True, replaceVal=1):
    dataframe = DataFrame.copy()
    dataframe[target] = dataframe[target].map(float)
    sysAvg = dataframe[dataframe[target] == 1].shape[0] / dataframe.shape[0]
    floatRe = re.compile('-?\d+\.+\d')
    try:
        if valueContinuous:
            dataframe[variable] = dataframe[variable].map(float)
            if equalWidth:
                dataframe['category'] = pd.cut(dataframe[variable], bins=bins)
            else:
                curQuantile = np.linspace(0, 1, bins + 1)
                bins = [round(dataframe[variable].quantile(x), 5) for x in curQuantile]
                bins = pd.Series(bins).unique()
                bins[0] = bins[0] - 0.1
                bins[-1] = bins[-1] + 0.1
                bins = pd.Series(bins).tolist()
                dataframe['category'] = pd.cut(dataframe[variable], bins=bins)
            dataByCategoryTarget = dataframe.groupby(['category', target])[target].count()
            ivDF = dataByCategoryTarget.unstack(target)
            if ivDF.isnull().sum().any():
                if replace:
                    ivDF = ivDF.fillna(replaceVal)
                else:
                    ivDFfill = ivDF
                    if ivDFfill.shape[0] <= 3:
                        ivDFfill = ivDFfill.fillna(replaceVal)
                    else:
                        while ivDFfill.isnull().sum().any():
                            nullRows = ivDFfill.isnull().sum(axis=1)
                            nrows = nullRows[nullRows > 0]
                            ivDFfill = ivDFfill.replace(np.NaN, 0)
                            for subIndexVal in nrows.index:
                                subIndex = nrows.index[nrows.index == subIndexVal]
                                if subIndex.isin(ivDFfill.index).any():
                                    try:
                                        subCanIndex = ivDFfill.index[np.where(ivDFfill.index == subIndexVal)[0] + 1]
                                        subRep = ivDFfill[ivDFfill.index.isin(subCanIndex)].values + ivDFfill[
                                            ivDFfill.index.isin(subIndex)].values
                                        subCanIndexPre = floatRe.findall(str(subCanIndex.values[0]))
                                        subIndexPre = floatRe.findall(str(subIndex.values[0]))
                                        subCanIndexPre.extend(subIndexPre)
                                        newSubIndexPre = pd.Series(pd.unique(subCanIndexPre)).map(float).sort_values()
                                        newSubIndex = pd.IntervalIndex.from_tuples(
                                            [(newSubIndexPre[0], newSubIndexPre[2])])
                                        subRep = pd.DataFrame(subRep, columns=ivDF.columns, index=newSubIndex)
                                        subDFfill = ivDFfill[~ivDFfill.index.isin(subCanIndex.append(subIndex))]
                                        subDFfill.index = pd.IntervalIndex(subDFfill.index)
                                        ivDFfill = subDFfill.append(subRep)
                                    except:
                                        subCanIndex = ivDFfill.index[np.where(ivDFfill.index == subIndexVal)[0] - 1]
                                        if len(subCanIndex.values) > 0:
                                            subRep = ivDFfill[ivDFfill.index.isin(subCanIndex)].values + ivDFfill[
                                                ivDFfill.index.isin(subIndex)].values
                                            subCanIndexPre = floatRe.findall(str(subCanIndex.values[0]))
                                            subIndexPre = floatRe.findall(str(subIndex.values[0]))
                                            subCanIndexPre.extend(subIndexPre)
                                            newSubIndexPre = pd.Series(pd.unique(subCanIndexPre)).map(
                                                float).sort_values()
                                            newSubIndex = pd.IntervalIndex.from_tuples(
                                                [(newSubIndexPre[0], newSubIndexPre[2])])
                                            subRep = pd.DataFrame(subRep, columns=ivDF.columns, index=newSubIndex)
                                            subDFfill = ivDFfill[~ivDFfill.index.isin(subCanIndex.append(subIndex))]
                                            subDFfill.index = pd.IntervalIndex(subDFfill.index)
                                            ivDFfill = subDFfill.append(subRep)
                                        else:
                                            pass
                                ivDFfill = ivDFfill.replace(0, np.nan)
                    ivDF = ivDFfill
        else:
            dataByCategoryTarget = dataframe.groupby([variable, target])[target].count()
            ivDF = dataByCategoryTarget.unstack(target)
            if ivDF.isnull().sum().any():
                if replace:
                    ivDF = ivDF.fillna(replaceVal)
                else:
                    ivDFfill = ivDF
                    if ivDFfill.shape[0] <= 3:
                        ivDFfill = ivDFfill.fillna(replaceVal)
                    else:
                        while ivDFfill.isnull().sum().any():
                            nullRows = ivDFfill.isnull().sum(axis=1)
                            nrows = nullRows[nullRows > 0]
                            ivDFfill = ivDFfill.fillna(0)
                            for i, subIndexVal in enumerate(nrows.index):
                                # print('{}:  {}'.format(i,subIndexVal))
                                subIndex = nrows.index[nrows.index == subIndexVal]
                                if subIndex.isin(ivDFfill.index).any():
                                    print('{}:  {}'.format(i, subIndexVal))
                                    try:
                                        subCanIndex = ivDFfill.index[np.where(ivDFfill.index == subIndexVal)[0] + 1]
                                        subRep = ivDFfill[ivDFfill.index.isin(subCanIndex)].values + ivDFfill[
                                            ivDFfill.index.isin(subIndex)].values
                                        newSubIndex = pd.Categorical(
                                            ','.join([str(subIndex.values[0]), str(subCanIndex.values[0])]))
                                        subRep = pd.DataFrame(subRep, columns=ivDF.columns, index=newSubIndex)
                                        subDFfill = ivDFfill[~ivDFfill.index.isin(subCanIndex.append(subIndex))]
                                        ivDFfill = subDFfill.append(subRep)
                                    except:
                                        subCanIndex = ivDFfill.index[np.where(ivDFfill.index == subIndexVal)[0] - 1]
                                        if len(subCanIndex.values) > 0:
                                            subRep = ivDFfill[ivDFfill.index.isin(subCanIndex)].values + ivDFfill[
                                                ivDFfill.index.isin(subIndex)].values
                                            newSubIndex = pd.Categorical(
                                                ','.join([str(subIndex.values[0]), str(subCanIndex.values[0])]))
                                            subRep = pd.DataFrame(subRep, columns=ivDF.columns, index=newSubIndex)
                                            subDFfill = ivDFfill[~ivDFfill.index.isin(subCanIndex.append(subIndex))]
                                            ivDFfill = subDFfill.append(subRep)
                                        else:
                                            pass
                                ivDFfill = ivDFfill.replace(0, np.nan)
                                print('current null num:{}'.format(ivDFfill.isnull().sum().any()))
                    ivDF = ivDFfill
        ivDF.columns = pd.Series(ivDF.columns).map({1: 'pos', 0: 'neg'})
        posNegSampSize = ivDF.apply(sum, axis=0)
        ivDF['negPert'] = ivDF['neg'] / posNegSampSize['neg']
        ivDF['posPert'] = ivDF['pos'] / posNegSampSize['pos']
        ivDF['cumNegPert'] = np.cumsum(ivDF['negPert'])
        ivDF['cumPosPert'] = np.cumsum(ivDF['posPert'])
        ivDF['lift'] = (ivDF['pos'] / (ivDF['pos'] + ivDF['neg'])) / sysAvg
        ivDF['woe'] = (ivDF['posPert'] / ivDF['negPert']).map(math.log)
        ivDF['woe'] = ivDF['woe'].map(lambda x: round(x, 6))
        ivDF['iv'] = (ivDF['posPert'] - ivDF['negPert']) * ivDF['woe']
        totalValue = ivDF.apply(sum, axis=0)
        ivDF2 = ivDF.copy()
        ivDF2 = ivDF2.append(totalValue, ignore_index=True)
        dfIndex = list(ivDF.index)
        dfIndex.append('all')
        ivDF2.index = dfIndex
        ivDF2.index.name = variable
        return (ivDF2)
    except:
        print('please check valueType or equal width equal frequency or bins')


def calIVAndTransform(DataFrame, continousCol, target, replace=False):
    dataframe = DataFrame.copy()
    woeRe = re.compile('-?\d+\.+\d+')
    featureCol = dataframe.columns[dataframe.columns != target]
    ivDict = {}
    continousBins = []
    continousWoe = []
    indexCol = []
    cateWoeDict = {}
    woeDF = pd.DataFrame(np.zeros((dataframe.shape[0], dataframe.shape[1])))
    woeDF.columns = dataframe.columns
    for i, subCol in enumerate(featureCol):
        print('current feaure {} : {} '.format(i, subCol))
        if subCol in continousCol:
            print('current data type : continuous')
            ivDF = ivAutoCombing(DataFrame, subCol, target, valueContinuous=True, replace=replace)
            subIVDF = ivDF.copy()
            subIVDF.index = range(ivDF.shape[0])
            subIVDict = dict(zip(subIVDF.index, subIVDF['woe']))
            subWOEBinsPre = woeRe.findall(str(ivDF.index))
            subWOEBins = pd.Series(pd.unique(subWOEBinsPre)).map(float).sort_values().tolist()
            continousBins.append(subWOEBins)
            continousWoe.append(ivDF['woe'][:-1].tolist())
            subData = dataframe[subCol]
            subDataBins = pd.cut(subData, subWOEBins, labels=list(range(ivDF.shape[0] - 1)))
            subDataWOE = subDataBins.map(subIVDict)
            # print('{} : {}  '.format(subCol,subWOEBins))
            woeDF[subCol] = subDataWOE
            indexCol.append(subCol)
        else:
            print('current data type : category')
            subColDict = {}
            subColCnts = dataframe[subCol].value_counts()
            subColNum = len(subColCnts)
            print(subCol, subColNum)
            if subColNum > 10:
                subColCntDesc = subColCnts.sort_values(ascending=False)
                curCol = subColCntDesc.index.values.tolist()
                subColDict.update(dict(zip(curCol[:9], curCol[:9])))
                subColDict.update(dict(zip(curCol[9:], ["other"] * len(curCol[9:]))))
                subData = dataframe[subCol].map(subColDict)
            else:
                subData = dataframe[subCol]
            subDataFrame = DataFrame.copy()
            subDataFrame[subCol] = subData
            ivDF = ivAutoCombing(subDataFrame, subCol, target, valueContinuous=False, replace=replace)
            subIVDF = ivDF[:-1]
            subIVDict = dict(zip(subIVDF.index, subIVDF['woe']))
            cateWoeDict.update({subCol: subIVDict})
            subWOEData = subDataFrame[subCol]
            subDataWOE = subWOEData.map(subIVDict)
            # print('subDataWOE: {}  '.format(subDataWOE[:5]))
            woeDF[subCol] = subDataWOE
        ivValue = ivDF.loc['all', 'iv']
        ivDict[subCol] = ivValue
    ivSeries = pd.Series(ivDict)
    ivSeries = ivSeries.sort_values(ascending=False)
    continousBinsDF = pd.DataFrame(continousBins)
    continousBinsDF.index = indexCol
    continousWoeDF = pd.DataFrame(continousWoe)
    continousWoeDF.index = indexCol
    return (ivSeries, woeDF, continousBinsDF, continousWoeDF, cateWoeDict)


def calDataframeIVAuto(DataFrame, continousCol, target, replace=False):
    dataframe = DataFrame.copy()
    featureCol = dataframe.columns[dataframe.columns != target]
    ivDict = {}
    for i, subCol in enumerate(featureCol):
        print('current feaure {} : {} '.format(i, subCol))
        if subCol in continousCol:
            print('current data type : continuous')
            ivDF = ivAutoCombing(DataFrame, subCol, target, valueContinuous=True, replace=replace)
        else:
            print('current data type : category')
            ivDF = ivAutoCombing(DataFrame, subCol, target, valueContinuous=False, replace=replace)
        ivValue = ivDF.loc['all', 'iv']
        ivDict[subCol] = ivValue
    ivSeries = pd.Series(ivDict)
    ivSeries = ivSeries.sort_values()
    return (ivSeries)


def getCorrThre(DataFrame, threhold=0.85):
    dataframe = DataFrame.copy()
    corrDict = {}
    featureName = dataframe.columns
    featureNum = len(featureName)
    corrDF = dataframe.corr()
    maxLen = 0
    for i, subCol in enumerate(featureName):
        # print('current feature {} : {}'.format(i,subCol))
        subCorr = corrDF[subCol]
        subCorrThre = subCorr[subCorr > threhold]
        subCorrFeature = [x for x in subCorrThre.index]
        subLen = len(subCorrFeature)
        maxLen = np.where(maxLen >= subLen, maxLen, subLen)
        corrDict[subCol] = subCorrFeature
    corrThreDF = pd.DataFrame(np.zeros((featureNum, maxLen + 1)))
    for i, item in enumerate(corrDict.items()):
        print('{}  : {}'.format(i, item))
        corrThreDF.iloc[i, 0] = item[0]
        subValue = sorted(item[1])
        for ii, value in enumerate(subValue):
            corrThreDF.iloc[i, ii + 1] = value
    corrThreDF = corrThreDF.sort_values(list(corrThreDF.columns))
    return (corrThreDF)


def get_noncollienar_col(colliear_col_df):
    keep_col_corr = []
    drop_col_corr = []
    for rownum in range(colliear_col_df.shape[0]):
        curr_row = colliear_col_df.iloc[rownum, :]
        curr_row = [item for item in curr_row if item != 0]
        if curr_row[0] not in drop_col_corr:
            keep_col_corr.append(curr_row[0])
        [drop_col_corr.append(item) for item in curr_row[2:] if item not in keep_col_corr]
        drop_col_corr = list(set(drop_col_corr))
    return (keep_col_corr, drop_col_corr)


def get_model_data_cont_cate(dataframe, sele_continuous_cols, sele_category_cols, target_col, scaler=False):
    standard_scaler = StandardScaler()
    sub_continuous_df = dataframe.loc[:, sele_continuous_cols]
    if scaler:
        sub_continuous_ndarray = standard_scaler.fit_transform(sub_continuous_df)
        sub_continuous_df = pd.DataFrame(sub_continuous_ndarray)
        sub_continuous_df.columns = sele_continuous_cols
    for sub_cate_col in sele_category_cols:
        sub_dummies_df = pd.get_dummies(dataframe.loc[:, sub_cate_col], prefix=sub_cate_col)
        sub_continuous_df = sub_continuous_df.join(sub_dummies_df.iloc[:, 1:], how='inner')
    sub_continuous_df = sub_continuous_df.join(dataframe.loc[:, target_col], how='inner')
    return (sub_continuous_df)


def show_ml_metric(test_labels, predict_labels, predict_prob):
    accuracy = accuracy_score(test_labels, predict_labels)
    precision = precision_score(test_labels, predict_labels)
    recall = recall_score(test_labels, predict_labels)
    f1_measure = f1_score(test_labels, predict_labels)
    confusionMatrix = confusion_matrix(test_labels, predict_labels)
    fpr, tpr, threshold = roc_curve(test_labels, predict_prob, pos_label=1)
    Auc = auc(fpr, tpr)
    MAP = average_precision_score(test_labels, predict_prob)
    print("------------------------- ")
    print("confusion matrix:")
    print("------------------------- ")
    print("| TP: %5d | FP: %5d |" % (confusionMatrix[1, 1], confusionMatrix[0, 1]))
    print("----------------------- ")
    print("| FN: %5d | TN: %5d |" % (confusionMatrix[1, 0], confusionMatrix[0, 0]))
    print(" ------------------------- ")
    print("Accuracy:       %.2f%%" % (accuracy * 100))
    print("Recall:         %.2f%%" % (recall * 100))
    print("Precision:      %.2f%%" % (precision * 100))
    print("F1-measure:     %.2f%%" % (f1_measure * 100))
    print("AUC:            %.2f%%" % (Auc * 100))
    print("MAP:            %.2f%%" % (MAP * 100))
    print("------------------------- ")
    return (Auc)


def getIntervalIndex(y_test, pred_proba):
    bins = np.arange(0, 1.1, 0.1)
    skPredDF = pd.DataFrame({'trueFlag': y_test, 'predProb': pred_proba})
    skPredDF['stratified'] = pd.cut(skPredDF['predProb'], bins=bins)
    skStratifiedDF = skPredDF.groupby(['stratified']).agg({'predProb': 'count', 'trueFlag': 'sum'})
    skStratifiedDF['precision'] = skStratifiedDF['trueFlag'] / skStratifiedDF['predProb']
    skStratifiedDF['precision'].plot(kind='bar')
    plt.title('测试集分段统计命中率')
    plt.show()
    skStratifiedDF2 = skStratifiedDF.sort_index(ascending=False)
    skStratifiedDF2 = skStratifiedDF2.fillna(0)
    skStratifiedDF2['cumul_pred'] = skStratifiedDF2['predProb'].cumsum()
    skStratifiedDF2['cumul_true'] = skStratifiedDF2['trueFlag'].cumsum()
    skStratifiedDF2['cumul_pred_rate'] = skStratifiedDF2['cumul_pred'] / skStratifiedDF2['predProb'].sum()
    skStratifiedDF2['cumul_precision'] = skStratifiedDF2['cumul_true'] / skStratifiedDF2['cumul_pred']
    skStratifiedDF2['cumul_recall'] = skStratifiedDF2['cumul_true'] / skStratifiedDF2['trueFlag'].sum()
    skStratifiedDF2['lift'] = skStratifiedDF2['precision'] / (
                skStratifiedDF2['trueFlag'].sum() / skStratifiedDF2['predProb'].sum())
    skStratifiedDF2['cumul_lift'] = skStratifiedDF2['cumul_precision'] / (
                skStratifiedDF2['trueFlag'].sum() / skStratifiedDF2['predProb'].sum())
    skStratifiedDF2 = skStratifiedDF2[
        ['predProb', 'trueFlag', 'precision', 'cumul_pred_rate', 'cumul_precision', 'cumul_recall', 'lift',
         'cumul_lift']]
    print(skStratifiedDF2)
    return (skStratifiedDF2)


def getTOPIndex(pred_proba, test_label, classNums=2):
    if classNums == 2:
        to_multi_df = pd.DataFrame({'predict_prob': pred_proba, 'test_label': test_label})
        to_multi_df = to_multi_df.sort_values(by='predict_prob', ascending=False)
        to_multi_df['order'] = to_multi_df['predict_prob'].rank(method='first', ascending=False)
        top_bins = [item * math.ceil(to_multi_df.shape[0] / 10) for item in range(1, 11, 1)]
        top_bins.insert(0, 0)
        to_multi_df['stratified'] = pd.cut(to_multi_df['order'], bins=top_bins)
        top_multi_df = to_multi_df.groupby(['stratified']).agg({'predict_prob': 'count', 'test_label': 'sum'})
        top_multi_df['cumul_test'] = top_multi_df['test_label'].cumsum()
        top_multi_df['cumul_pred'] = top_multi_df['predict_prob'].cumsum()
        top_multi_df['precision'] = top_multi_df['test_label'] / top_multi_df['predict_prob']
        top_multi_df['cumul_precision'] = top_multi_df['cumul_test'] / top_multi_df['cumul_pred']
        top_multi_df['cumul_recall'] = top_multi_df['cumul_test'] / top_multi_df['test_label'].sum()
        top_multi_df['lift'] = top_multi_df['precision'] / (
                    top_multi_df['test_label'].sum() / top_multi_df['predict_prob'].sum())
        top_multi_df['cumul_lift'] = top_multi_df['cumul_precision'] / (
                    top_multi_df['test_label'].sum() / top_multi_df['predict_prob'].sum())
        top_result_multi_df = top_multi_df[['predict_prob', 'test_label', 'precision',
                                            'cumul_precision', 'cumul_recall', 'lift', 'cumul_lift']]
        print(top_result_multi_df)
        return (top_result_multi_df)
    else:
        to_multi_df = pd.DataFrame(pred_proba, columns=list(range(0, classNums)))
        subClass = to_multi_df.columns
        to_multi_df['test_label'] = test_label
        init_test_label = [1 if item == int(subClass[0]) else 0 for item in test_label]
        sub_top_result_multi_df = getTOPIndex(to_multi_df[0], init_test_label, 2)
        sub_top_result_multi_df.index = [list(str(subClass[0]) * 10), sub_top_result_multi_df.index]
        for subClas in subClass[1:]:
            sub_test_label = [1 if item == int(subClas) else 0 for item in test_label]
            sub_top_result_multi_df2 = getTOPIndex(to_multi_df[subClas], sub_test_label, 2)
            sub_top_result_multi_df2.index = [list(str(subClas) * 10), sub_top_result_multi_df2.index]
            sub_top_result_multi_df = sub_top_result_multi_df.append(sub_top_result_multi_df2)
        print(sub_top_result_multi_df)
        return (sub_top_result_multi_df)


def getAUC(predVal, trueFlag, threNum=100):
    threArr = np.linspace(0, 1, threNum)
    aucMat = np.zeros((len(threArr), 4))
    for i, curThre in enumerate(threArr):
        predFlag = np.where(predVal > curThre, 1, 0)
        predFlagCate = pd.Categorical(predFlag, categories=[0, 1])
        trueFlagCate = pd.Categorical(trueFlag, categories=[0, 1])
        curConfusionMatrix = pd.crosstab(trueFlagCate, predFlagCate, dropna=False)
        if sum(curConfusionMatrix.iloc[1, :]) == 0:
            TPR = 0
        else:
            TPR = curConfusionMatrix.iloc[1, 1] / sum(curConfusionMatrix.iloc[1, :])
        if sum(curConfusionMatrix.iloc[0, :]) == 0:
            FPR == 0
        else:
            FPR = curConfusionMatrix.iloc[0, 1] / sum(curConfusionMatrix.iloc[0, :])
        if sum(curConfusionMatrix.iloc[:, 1]) == 0:
            PR == 0
        else:
            PR = curConfusionMatrix.iloc[1, 1] / sum(curConfusionMatrix.iloc[:, 1])
        aucMat[i] = FPR, TPR, PR, curThre
    aucDF = pd.DataFrame(aucMat, columns=['FPR', 'TPR', 'PR', 'threhold'])
    return (aucDF)


def get_linear_metric(model_y_test, model_y_pred, error_rate=0.2):
    R2 = r2_score(model_y_test, model_y_pred)
    MSE = mean_squared_error(model_y_test, model_y_pred)
    RMSE = np.sqrt(MSE)
    MAE = mean_absolute_error(model_y_test, model_y_pred)
    error_list = [True if (x >= (y - y * error_rate)) and (x <= (y + y * error_rate)) else False for x, y in
                  zip(model_y_pred, model_y_test)]
    Error_ACC = sum(error_list) / len(error_list)
    return (R2, MSE, RMSE, MAE, Error_ACC)
    print("model metric \nR2: {} \nMSE: {} \nRMSE： {} \nMAE： {} \nError_ACC： {}".format(R2, MSE, RMSE, MAE, Error_ACC))


def get_feat_imp_thre(list_columns, imp_values, sumImp=False, square_threhold=0.5, max_threhold=0.4):
    imp_df = pd.DataFrame({"feat_name": list_columns, "feat_values": imp_values})
    imp_df = imp_df[["feat_name", "feat_values"]]
    imp_df = imp_df.sort_values("feat_values", ascending=False)
    imp_df["values_square"] = imp_df["feat_values"].map(lambda x: math.pow(x, 2))
    imp_df["values_square_cumu"] = imp_df["values_square"].cumsum()
    imp_df["values_square_rate"] = imp_df["values_square_cumu"] / imp_df["values_square_cumu"].max()
    imp_df["values_rate"] = imp_df["feat_values"] / imp_df["feat_values"].max()
    return_feat = None
    if sumImp:
        selectedFeatureSquare = imp_df["feat_name"][imp_df["values_square_rate"] <= square_threhold]
        print("selected feature by squres sum: %d" % (len(selectedFeatureSquare)))
        return_feat = selectedFeatureSquare.to_list()
    else:
        selectedFeatureMax = imp_df["feat_name"][imp_df["values_rate"] >= max_threhold]
        print("selected feature by max: %d" % (len(selectedFeatureMax)))
        return_feat = selectedFeatureMax.to_list()
    if len(return_feat) == 0:
        return_feat = imp_df["feat_name"].head(25).to_list()
    return (return_feat, imp_df.loc[:, ["feat_name", "feat_values"]])


def draw_ROC(skAuc, areaAucVal, save_fig_path=None):
    ''' input:  @skAUC       : DataFrame,containg two columns FPR,TPR, respectively
                @areaAucVal  : float,auc value
    '''
    figure = plt.figure()
    ax = figure.add_subplot(1, 1, 1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.plot(skAuc['FPR'], skAuc['TPR'], color='red', linestyle='-', linewidth=1, label='AUC')
    # ax.text(0.2,0.85,'auc area: %f.2' %areaAucVal)
    ax.annotate('auc area: %f.4' % areaAucVal, xy=(areaAucVal - 0.3, areaAucVal - 0.15), xycoords='data',
                xytext=(areaAucVal - 0.2, areaAucVal - 0.2),
                arrowprops=dict(facecolor='blue', width=1.3, shrink=1, connectionstyle="arc3,rad=.2"))
    ax.set_xlabel('FPR')
    ax.set_ylabel('召回率')
    ax.set_title('auc of logistic')
    ax.legend(loc='upper left')
    if save_fig_path:
        plt.savefig(save_fig_path)
    plt.show()


def draw_TPR_PR_thre(skAuc, save_fig_path=None):
    ''' input:  @skAUC       : DataFrame,containg two columns FPR,TPR, respectively
                @areaAucVal  : float,auc value
    '''
    figure = plt.figure()
    ax = figure.add_subplot(1, 1, 1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.plot(skAuc['threhold'], skAuc['TPR'], color='red', linestyle='--', marker='o', linewidth=0.7, label='召回率')
    ax.plot(skAuc['threhold'], skAuc['PR'], color='blue', linestyle='-', marker='o', linewidth=0.7, label='命中率')
    ax.set_xlabel('阈值')
    ax.set_ylabel('召回率和命中率')
    ax.set_title('tpr&pr with threhold')
    ax.legend(loc='upper left')
    if save_fig_path:
        plt.savefig(save_fig_path)
    plt.show()


####################################     测试集分段统计人数
def cal_prob_interal_nums(pred_proba, save_fig_path=None):
    bins = np.arange(0, 1.1, 0.1)
    yrstSeries = pd.cut(pred_proba, bins=bins)
    yScoreSeg = yrstSeries.value_counts().sort_index()
    yScoreSeg.plot(kind='bar')
    plt.title('测试集分段统计')
    if save_fig_path:
        plt.savefig(save_fig_path)
    plt.show()
    print(yScoreSeg)


####################################     预测多分类问题结果汇总
def getMultiResult(pred_multi_label, pred_multi_proba, test_label):
    numClass = len(set(pred_multi_label))
    pred_multi_binary_df = pd.DataFrame(pred_multi_proba, columns=list(range(0, numClass)))
    return_multi_binary_proba = pred_multi_binary_df.max(axis=1)
    return_binary_label = [1 if x == y else 0 for x, y in zip(test_label, pred_multi_label)]
    return (return_multi_binary_proba, return_binary_label)


##########################################################################################################################################
def eval_model(modeler, model_feature, model_target, model_type_reg=True, tune=False, tune_dict=None,
               base_estimator_tree=True, tune_sele_feat=False, cross_feature=False,
               output_path='/home/changtou/Changtou/CT_work/Market'):
    ''' input:
              @model_feature  :DataFrame,containing colummns
              @model_target   :DataFrame or Series
              @model_type_reg :boolean, check wether modeler is a regressor or not ,default True
              @tune           :boolean, check wether modeler needs to be tuned or not ,default False
              @cross_feature  :boolean, check wether modeler needs to cross features by polynomial ,default False
              @tune_dict      :dict,for instance ；tune_dict={'cv':5,'scoring':'mse','param_dict':{'max_depth':[8],'min_samples_leaf':[50]}}
    '''
    model_x_train, model_x_test, model_y_train, model_y_test = train_test_split(model_feature, model_target,
                                                                                test_size=0.25, random_state=666)
    if model_type_reg:
        sub_modeler = modeler
        sub_modeler.fit(model_x_train, model_y_train)
    else:
        print('after Categorized preprocessing result: \n{} '.format(pd.Series(model_y_train).value_counts()))
        ros = RandomOverSampler(random_state=666)
        x_resampled, y_resampled = ros.fit_sample(model_x_train, model_y_train)
        x_resampled_df = pd.DataFrame(x_resampled)
        x_resampled_df.columns = model_feature.columns
        model_x_train = x_resampled_df
        y_resampled = [int(x) for x in y_resampled]
        model_y_train = pd.Series(y_resampled)
        print('after ROS preprocessing result: \n{} '.format(model_y_train.value_counts()))
        sub_modeler = modeler
        sub_modeler.fit(model_x_train, model_y_train)
    sub_feat_cols = model_feature.columns.tolist()
    sub_select_molder = sub_modeler
    if model_type_reg:
        if base_estimator_tree:
            sub_imp_values = sub_modeler.feature_importances_
            sub_imp_sele_feat, sub_imp_df = get_feat_imp_thre(sub_feat_cols, sub_imp_values)
            print(''.join(['*'] * 20 + ['coefficients df'] + ['*'] * 20 + ['\n']))
            print(sub_imp_df.head(25))
        else:
            coef_list = sub_modeler.coef_.reshape(-1).tolist()
            coef_list.append(sub_modeler.intercept_)
            feat_list = sub_feat_cols.copy()
            feat_list.append("intercept")
            coef_df = pd.DataFrame({"feat": feat_list, "coef": coef_list}, index=range(len(coef_list)))
            coef_df = coef_df.sort_values('coef', ascending=False)
            coef_df = coef_df.loc[:, ['feat', 'coef']]
            print(''.join(['*'] * 20 + ['coefficients df'] + ['*'] * 20 + ['\n']))
            print(coef_df.head(25))
        if tune_sele_feat:
            x_train_sele_df = model_x_train.loc[:, sub_imp_sele_feat]
            x_test_sele_df = model_x_test.loc[:, sub_imp_sele_feat]
            sub_modeler = modeler
            sub_modeler.fit(x_train_sele_df, model_y_train)
        else:
            x_train_sele_df = model_x_train
            x_test_sele_df = model_x_test
        if cross_feature:
            polyfeature = PolynomialFeatures(degree=2)
            x_train_poly_array = polyfeature.fit_transform(x_train_sele_df)
            x_train_poly_df = pd.DataFrame(x_train_poly_array)
            sub_poly_col = ['poly_' + str(i) for i in range(x_train_poly_df.shape[1])]
            x_train_poly_df.columns = sub_poly_col
            x_train_sele_df = x_train_poly_df
            x_test_poly_array = polyfeature.transform(x_test_sele_df)
            x_test_poly_df = pd.DataFrame(x_test_poly_array)
            x_test_poly_df.columns = sub_poly_col
            x_test_sele_df = x_test_poly_df
            sub_modeler = modeler
            sub_modeler.fit(x_train_sele_df, model_y_train)
        try:
            sub_pred = sub_modeler.predict(x_test_sele_df)
        except:
            sub_pred = sub_modeler.predict(model_x_test)
        R2, MSE, RMSE, MAE, Error_ACC = get_linear_metric(model_y_test, sub_pred)
        print(''.join(['*'] * 20 + ['   raw model metrics    '] + ['*'] * 20 + ['\n']))
        print("raw regressor model metric \nR2: {} \nMSE: {} \nRMSE： {} \nMAE： {} \nError_ACC: {}".format(R2, MSE, RMSE,
                                                                                                          MAE,
                                                                                                          Error_ACC))
        if tune:
            sub_scoring = tune_dict['scoring']
            sub_cv = tune_dict['cv']
            sub_params = tune_dict['param_dict']
            regressor_gs = GridSearchCV(estimator=modeler, param_grid=sub_params, cv=sub_cv, scoring=sub_scoring,
                                        random_state=666)
            regressor_gs.fit(model_x_train, model_y_train)
            print(''.join(['*'] * 20 + ['    tuning modeler    '] + ['*'] * 20 + ['\n']))
            print('grid search results: {}\n'.format(regressor_gs.cv_results_))
            print('grid search socres: {}\n'.format(regressor_gs.best_score_))
            print('grid search socres: {}\n'.format(regressor_gs.best_params_))
            sub_gs_pred = regressor_gs.best_estimator_.predict(model_x_test)
            R2, MSE, RMSE, MAE, Error_ACC = get_linear_metric(model_y_test, sub_gs_pred)
            print(
                "regressor model metric by grid search \nR2: {} \nMSE: {} \nRMSE： {} \nMAE： {} \nError_ACC: {}".format(
                    R2, MSE, RMSE, MAE, Error_ACC))
            sub_select_molder = regressor_gs.best_estimator_
        return (sub_select_molder)
    else:
        if base_estimator_tree:
            sub_imp_values = sub_modeler.feature_importances_
            sub_imp_sele_feat, sub_imp_df = get_feat_imp_thre(sub_feat_cols, sub_imp_values)
            print(''.join(['*'] * 20 + ['impor df'] + ['*'] * 20 + ['\n']))
            print(sub_imp_df.head(25))
        else:
            coef_list = sub_modeler.coef_.reshape(-1).tolist()
            coef_list.append(sub_modeler.intercept_)
            feat_list = sub_feat_cols.copy()
            feat_list.append("intercept")
            coef_df = pd.DataFrame({"feat": feat_list, "coef": coef_list}, index=range(len(coef_list)))
            coef_df = coef_df.sort_values('coef', ascending=False)
            coef_df = coef_df.loc[:, ['feat', 'coef']]
            print(''.join(['*'] * 20 + ['coefficients df'] + ['*'] * 20 + ['\n']))
            print(coef_df.head(25))
        if tune_sele_feat:
            x_train_sele_df = model_x_train.loc[:, sub_imp_sele_feat]
            x_test_sele_df = model_x_test.loc[:, sub_imp_sele_feat]
            sub_select_molder = modeler
            sub_select_molder.fit(x_train_sele_df, model_y_train)
        else:
            x_train_sele_df = model_x_train
            x_test_sele_df = model_x_test
        if cross_feature:
            polyfeature = PolynomialFeatures(degree=2)
            x_train_poly_array = polyfeature.fit_transform(x_train_sele_df)
            x_train_poly_df = pd.DataFrame(x_train_poly_array)
            sub_poly_col = ['poly_' + str(i) for i in range(x_train_poly_df.shape[1])]
            x_train_poly_df.columns = sub_poly_col
            x_train_sele_df = x_train_poly_df
            x_test_poly_array = polyfeature.transform(x_test_sele_df)
            x_test_poly_df = pd.DataFrame(x_test_poly_array)
            x_test_poly_df.columns = sub_poly_col
            x_test_sele_df = x_test_poly_df
            sub_select_molder = modeler
            sub_select_molder.fit(x_train_sele_df, model_y_train)
        if tune:
            sub_scoring = tune_dict['scoring']
            sub_cv = tune_dict['cv']
            sub_params = tune_dict['param_dict']
            classifier_gs = GridSearchCV(estimator=modeler, param_grid=sub_params, cv=sub_cv, scoring=sub_scoring)
            classifier_gs.fit(x_train_sele_df, model_y_train)
            print(''.join(['*'] * 20 + ['    tuning modeler    '] + ['*'] * 20 + ['\n']))
            print('grid search results: {}\n'.format(classifier_gs.cv_results_))
            print('grid search socres: {}\n'.format(classifier_gs.best_score_))
            print('grid search socres: {}\n'.format(classifier_gs.best_params_))
            sub_select_molder = classifier_gs.best_estimator_
        try:
            sub_gs_pred_class = sub_select_molder.predict(model_x_test)
            sub_gs_pred_proba = sub_select_molder.predict_proba(model_x_test)
        except:
            sub_gs_pred_class = sub_select_molder.predict(x_test_sele_df)
            sub_gs_pred_proba = sub_select_molder.predict_proba(x_test_sele_df)
        class_nums = len(pd.Series(sub_gs_pred_class).unique())
        if class_nums == 2:
            sub_gs_pred_proba = sub_gs_pred_proba[:, 1]
            print(''.join(['*'] * 20 + ['    binary metrics    '] + ['*'] * 20 + ['\n']))
            sub_auc = show_ml_metric(model_y_test, sub_gs_pred_class, sub_gs_pred_proba)
            print(''.join(['*'] * 20 + ['    binary ROC    '] + ['*'] * 20 + ['\n']))
            skAUC = getAUC(sub_gs_pred_proba, model_y_test, threNum=100)
            draw_ROC(skAUC, sub_auc)
            print(''.join(['*'] * 20 + ['   TPR&FPR    '] + ['*'] * 20 + ['\n']))
            draw_TPR_PR_thre(skAUC)
            print(''.join(['*'] * 20 + ['    prob_interal_nums    '] + ['*'] * 20 + ['\n']))
            cal_prob_interal_nums(sub_gs_pred_proba)
            print(''.join(['*'] * 20 + ['    INTERVAL INDEX    '] + ['*'] * 20 + ['\n']))
            sub_interval_index = getIntervalIndex(model_y_test, sub_gs_pred_proba)
            filename = 'binary_interval_index_' + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + '.csv'
            sub_interval_index.to_csv(os.path.join(output_path, filename), encoding='utf-8')
            print(''.join(['*'] * 20 + ['    TOP INDEX    '] + ['*'] * 20 + ['\n']))
            sub_top_index = getTOPIndex(sub_gs_pred_proba, model_y_test, classNums=2)
            filename = 'binary_top_index_' + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + '.csv'
            sub_top_index.to_csv(os.path.join(output_path, filename), encoding='utf-8')
        else:
            multi_binary_proba, multi_binary_label = getMultiResult(sub_gs_pred_class, sub_gs_pred_proba, model_y_test)
            multi_confusion_mat = confusion_matrix(model_y_test, sub_gs_pred_class)
            print(''.join(['*'] * 20 + ['   multi confusion matrix   '] + ['*'] * 20 + ['\n']))
            print(pd.DataFrame(multi_confusion_mat))
            print(''.join(['*'] * 20 + ['   multi classification report  '] + ['*'] * 20 + ['\n']))
            print(classification_report(model_y_test, sub_gs_pred_class))
            print(''.join(['*'] * 20 + ['   prob_interal_nums  '] + ['*'] * 20 + ['\n']))
            cal_prob_interal_nums(multi_binary_proba)
            print(''.join(['*'] * 20 + ['    INTERVAL INDEX    '] + ['*'] * 20 + ['\n']))
            sub_multi_interval_index = getIntervalIndex(multi_binary_label, multi_binary_proba)
            filename = 'multi_interval_index_' + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + '.csv'
            sub_multi_interval_index.to_csv(os.path.join(output_path, filename), encoding='utf-8')
            print(''.join(['*'] * 20 + ['    TOP INDEX    '] + ['*'] * 20 + ['\n']))
            sub_multi_top_index = getTOPIndex(sub_gs_pred_proba, model_y_test, classNums=class_nums)
            filename = 'multi_top_index_' + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + '.csv'
            sub_multi_top_index.to_csv(os.path.join(output_path, filename), encoding='utf-8')
        return (sub_select_molder)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt


def get_bool_norms(list):
    if type(list) == np.ndarray:
        array = list
    else:
        array = np.array(list)
    q1 = np.percentile(array, 25)
    q3 = np.percentile(array, 75)
    IQR = q3 - q1
    low_boundary = q1 - 1.5 * IQR
    up_boundary = q3 + 1.5 * IQR
    bool_list = [True if (low_boundary < x and x < up_boundary) else False for x in list]
    return (bool_list)


def plot_roc(y_test, y_score, num_class=[0, 1], label_name=[]):
    n_classes = y_test.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        print(i)
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(num_class, colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(label_name[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()





















































































