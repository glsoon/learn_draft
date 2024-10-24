#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/25 07:46
# @Author  : Gongle
# @File    : ml_pre.py
# @Version : 1.0
# @Desc    : None


ML - -preprocessing

1.
将多类转换为top类，其余归为other。

def get_less_cate(dataframe, col, top_num=10):
    subColDict = {}
    sub_data = dataframe[col]
    sub_data = sub_data.fillna('unknown')
    subColCnts = sub_data.value_counts()
    subColNum = len(subColCnts)
    print(col, subColNum)
    if subColNum > top_num:
        subColCntDesc = subColCnts.sort_values(ascending=False)
        curCol = subColCntDesc.index.values.tolist()
        subColDict.update(dict(zip(curCol[:top_num], curCol[:top_num])))
        subColDict.update(dict(zip(curCol[top_num:], ["other"] * len(curCol[top_num:]))))
    subData = sub_data.map(subColDict)
    return subData


2.
连续值和离散经过哑变量处理后合并形成最终模型数值框。
model_data_embed = model_data_pre.loc[:, embed_sele_cols]
sele_cate_col2 = [x for x in embed_sele_cols if x in sele_cate_col]
model_feature_data_onehot2 = model_data_embed.copy()
model_feature_data_onehot2 = model_feature_data_onehot2.drop(sele_cate_col2, axis=1)

for sub_feat in sele_cate_col2:
    sub_feat_dummy = pd.get_dummies(model_data_embed.loc[:, sub_feat], prefix=sub_feat)
    model_feature_data_onehot2 = model_feature_data_onehot2.join(sub_feat_dummy, how='inner')

3.
计算时间。

def get_time_i(tstamp):
    if pd.isnull(tstamp) or tstamp <= 0:
        return None
    else:
        tstamp = int(tstamp)
        if len(str(tstamp)) == 13:
            return (datetime.fromtimestamp(int(tstamp / 1000)))
        elif len(str(tstamp)) == 10:
            return (datetime.fromtimestamp(tstamp))
        else:
            None


def get_time_s(tstamp):
    if pd.isnull(tstamp) or len(tstamp) <= 0:
        return None
    else:
        tstamp = str(tstamp)
        if len(tstamp) == 10:
            return (datetime.strptime(tstamp, '%Y-%m-%d'))
        else:
            tstamp = tstamp[:19]
            return (datetime.strptime(tstamp, '%Y-%m-%d %H:%M:%S'))


def get_date_delta(ds1, ds2, mode='day'):
    ds_delta = ds2 - ds1
    ds_delta_day = ds_delta.days
    if mode == 'day':
        return ds_delta_day
    elif mode == 'hour':
        return int(ds_delta.seconds / 3600)
    else:
        return ds_delta


def get_delta_days(ds1, ds2):
    ds_delta = ds2.date() - ds1.date()
    ds_delta_day = ds_delta.days
    return ds_delta_day


def get_delta_diff(datetime_list, mode='second'):
    datetime_list = pd.Series(datetime_list).drop_duplicates().sort_values().to_list()
    datetime_interval_seconds = [(datetime_list[i + 1] - datetime_list[i]).seconds for i in
                                 range(len(datetime_list) - 1)]
    if mode == 'hour':
        return [round(item / 3600, 2) for item in datetime_interval_seconds]
    elif mode == 'minute':
        return [round(item / 60, 2) for item in datetime_interval_seconds]
    else:
        return datetime_interval_seconds


3.
统计数据分布。

def get_dist(value_dist, index_list=['min', '10%', '25%', '50%', 'mean', '75%', '90%', 'max', 'std'], precision=2):
    data_dist = pd.Series(value_dist).describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    sub_index = pd.Index(index_list)
    data_interval_dist_value = data_dist[sub_index].map(lambda x: round(x, precision)).to_list()
    return (data_interval_dist_value)


3.
模型内存不足进行预测。

def model_predict_step(model, model_data_df):
    model_pred_label = []
    model_pred_proba = []
    step = 10
    nums = model_data_df.shape[0]
    interval = int(nums / step) + 1
    for sub_step in range(step):
        start_index = interval * sub_step
        end_index = min(interval * sub_step + interval, nums)
        print('start: {:>7}   end: {:>7}'.format(start_index, end_index))
        sub_model_data = model_data_df.iloc[start_index:end_index, :]
        sub_pred_label = model.predict(sub_model_data)
        sub_pred_proba = model.predict_proba(sub_model_data)[:, 1]
        model_pred_label.extend(sub_pred_label.tolist())
        model_pred_proba.extend(sub_pred_proba.tolist())
    return model_pred_label, model_pred_proba


def k_model_predict_step(model, model_data_df):
    model_pred_label = []
    step = 10
    nums = model_data_df.shape[0]
    interval = int(nums / step) + 1
    for sub_step in range(step):
        start_index = interval * sub_step
        end_index = min(interval * sub_step + interval, nums)
        print('start: {:>7}   end: {:>7}'.format(start_index, end_index))
        sub_model_data = model_data_df.iloc[start_index:end_index, :]
        sub_pred_label = model.predict(sub_model_data)
        model_pred_label.extend(sub_pred_label.tolist())
    return model_pred_label


4.
确定好模型，训练特征和目标，测试特征和目标。

def train_model_clf_tree(modeler, model_x_train, model_y_train, model_x_test, model_y_test, output_path=os.getcwd()):
    print('after Categorized preprocessing result: \n{} '.format(pd.Series(model_y_train).value_counts()))
    ros = RandomOverSampler(random_state=666)
    x_resampled, y_resampled = ros.fit_sample(model_x_train, model_y_train)
    x_resampled_df = pd.DataFrame(x_resampled)
    x_resampled_df.columns = model_x_train.columns
    model_x_train = x_resampled_df
    y_resampled = [int(x) for x in y_resampled]
    model_y_train = pd.Series(y_resampled)
    print('after ROS preprocessing result: \n{} '.format(model_y_train.value_counts()))
    sub_modeler = modeler
    sub_modeler.fit(model_x_train, model_y_train)
    sub_feat_cols = model_x_train.columns.tolist()
    sub_select_molder = sub_modeler
    sub_imp_values = sub_modeler.feature_importances_
    sub_imp_sele_feat, sub_imp_df = ml_util.get_feat_imp_thre(sub_feat_cols, sub_imp_values)
    print(''.join(['*'] * 20 + ['impor df'] + ['*'] * 20 + ['\n']))
    print(sub_imp_df.head(25))
    x_train_sele_df = model_x_train.loc[:, sub_imp_sele_feat]
    x_test_sele_df = model_x_test.loc[:, sub_imp_sele_feat]
    sub_select_molder.fit(x_train_sele_df, model_y_train)
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
        sub_auc = ml_util.show_ml_metric(model_y_test, sub_gs_pred_class, sub_gs_pred_proba)
        print(''.join(['*'] * 20 + ['    binary ROC    '] + ['*'] * 20 + ['\n']))
        skAUC = ml_util.getAUC(sub_gs_pred_proba, model_y_test, threNum=100)
        ml_util.draw_ROC(skAUC, sub_auc)
        print(''.join(['*'] * 20 + ['   TPR&FPR    '] + ['*'] * 20 + ['\n']))
        ml_util.draw_TPR_PR_thre(skAUC)
        print(''.join(['*'] * 20 + ['    prob_interal_nums    '] + ['*'] * 20 + ['\n']))
        ml_util.cal_prob_interal_nums(sub_gs_pred_proba)
        print(''.join(['*'] * 20 + ['    INTERVAL INDEX    '] + ['*'] * 20 + ['\n']))
        sub_interval_index = ml_util.getIntervalIndex(model_y_test, sub_gs_pred_proba)
        filename = 'binary_interval_index_' + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + '.csv'
        sub_interval_index.to_csv(os.path.join(output_path, filename), encoding='utf-8')
        print(''.join(['*'] * 20 + ['    TOP INDEX    '] + ['*'] * 20 + ['\n']))
        sub_top_index = ml_util.getTOPIndex(sub_gs_pred_proba, model_y_test, classNums=2)
        filename = 'binary_top_index_' + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + '.csv'
        sub_top_index.to_csv(os.path.join(output_path, filename), encoding='utf-8')
    else:
        multi_binary_proba, multi_binary_label = ml_util.getMultiResult(sub_gs_pred_class, sub_gs_pred_proba,
                                                                        model_y_test)
        multi_confusion_mat = ml_util.confusion_matrix(model_y_test, sub_gs_pred_class)
        print(''.join(['*'] * 20 + ['   multi confusion matrix   '] + ['*'] * 20 + ['\n']))
        print(pd.DataFrame(multi_confusion_mat))
        print(''.join(['*'] * 20 + ['   multi classification report  '] + ['*'] * 20 + ['\n']))
        print(classification_report(model_y_test, sub_gs_pred_class))
        print(''.join(['*'] * 20 + ['   prob_interal_nums  '] + ['*'] * 20 + ['\n']))
        ml_util.cal_prob_interal_nums(multi_binary_proba)
        print(''.join(['*'] * 20 + ['    INTERVAL INDEX    '] + ['*'] * 20 + ['\n']))
        sub_multi_interval_index = ml_util.getIntervalIndex(multi_binary_label, multi_binary_proba)
        filename = 'multi_interval_index_' + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + '.csv'
        sub_multi_interval_index.to_csv(os.path.join(output_path, filename), encoding='utf-8')
        print(''.join(['*'] * 20 + ['    TOP INDEX    '] + ['*'] * 20 + ['\n']))
        sub_multi_top_index = ml_util.getTOPIndex(sub_gs_pred_proba, model_y_test, classNums=class_nums)
        filename = 'multi_top_index_' + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + '.csv'
        sub_multi_top_index.to_csv(os.path.join(output_path, filename), encoding='utf-8')
    return (sub_select_molder)