
'''
A5	计时点t0	搅拌并加热
A6	初始温度T0
A9	测温时刻t2
A10	实时温度T2	接近沸点
A11	计时点t3	水解开始，控制温度 A9到A11升温时间
A12	实时温度T3
A14	测温时刻t4	A11到A14缓慢升温时间
A15	实时温度T4	A10到A15缓慢升高的温度
A16	测温时刻t5	t5-t2=2h
A17	实时温度T5      A15到A17水解过程中温度的变化，可能升高也可能降低
A20	补水过程S6	30-60min
A21	脱色原料1
A22	脱色原料2
A24	脱色开始时间t7
A25	脱色开始温度T7
A26	脱色保温完成时间t8	    A24到A26脱色用时
A27	保温结束温度T8	A25到A27脱色过程中温度的变化
A28	去除脱色物质过程S9	 取时长

B1	神秘物质	可能是盐酸，B程序可能是结晶甩滤
B2	上述物质的浓度
B4	滴加过程S10	取时长
B5	滴加结束时间t11	似乎和B4重复
B6	滴加结束温度T11	A27到B6温度的变化
B7	测温时刻t12	B5到B7降温用时
B8	实时温度T12	B6到B8降温的幅度
B9	甩滤过程S13
B10	甩滤过程S14	S14,S15似乎可以合并
B12	滴加物质	作用于S13-S15，400/90min
B14	神秘物质	可能是纯化水，完成
'''

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy import sparse
import warnings
import time
import sys
import os
import re
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
from sklearn.metrics import mean_squared_error

def get_drop_speed(V, t):
    try:
        return V / t
    except:
        return 0

def id_features(s):
    return float(s.split('_')[1])

def minus_temperature(t1, t2):
    return t2 - t1


def TimeTransHours(StartTime, EndTime, DefaultHours):
    try:
        StartHour, StartMinute, _ = StartTime.split(':')
        EndHour, EndMinute, _ = EndTime.split(':')
        Hours = (int(EndHour) * 60 + int(EndMinute) - int(StartMinute) - int(
            StartHour) * 60) / 60.0
        Hours = Hours % 24
    except:
        Hours = DefaultHours
    return Hours


def GetTimeFeatures(se, DefaultHours):
    try:
        sh, sm, eh, em = re.findall(r"\d+\.?\d*", se)
    except:
        return DefaultHours

    try:
        if int(sh) > int(eh):
            tm = (int(eh) * 3600 + int(em) * 60 - int(sm) * 60 - int(sh) * 3600) / 3600.0 + 24
        else:
            tm = (int(eh) * 3600 + int(em) * 60 - int(sm) * 60 - int(sh) * 3600) / 3600.0
    except:
        return DefaultHours

    return tm


def FeatureEngineering(train):

    # 删除类别唯一的特征
    train.drop(['A1', 'A4', 'A8', 'A13', 'A16', 'A18', 'A23', 'B3'], axis=1, inplace=True)

    # 填充空缺特征
    train.fillna(0, inplace=True)

    # 转换时间特征
    train['A20'] = train.apply(lambda df: GetTimeFeatures(df['A20'], DefaultHours=0.0), axis=1)
    # train['A24'] = train.apply(lambda df: GetTimeFeatures(df['A24'], DefaultHours=0.0), axis=1)
    train['A28'] = train.apply(lambda df: GetTimeFeatures(df['A28'], DefaultHours=0.0), axis=1)
    train['B4'] = train.apply(lambda df: GetTimeFeatures(df['B4'], DefaultHours=0.0), axis=1)
    train['B9'] = train.apply(lambda df: GetTimeFeatures(df['B9'], DefaultHours=0.001), axis=1)
    train['B10'] = train.apply(lambda df: GetTimeFeatures(df['B10'], DefaultHours=0.001), axis=1)
    train['B11'] = train.apply(lambda df: GetTimeFeatures(df['B11'], DefaultHours=0.001), axis=1)

    # 添加时长特征
    train['A9'] = train.apply(lambda df: TimeTransHours(df['A9'], '00:00:00', DefaultHours=12.0), axis=1)

    # 删除无用的时间特征
    train.drop(['A5', 'A7', 'A24', 'A11', 'A14', 'A26', 'B5', 'B7'], axis=1, inplace=True)

    # 添加温度变化特征
    train['delta_temprature_1'] = train.apply(lambda df: minus_temperature(df['A10'], df['A15']), axis=1)
    train['delta_temprature_2'] = train.apply(lambda df: minus_temperature(df['A15'], df['A17']), axis=1)
    train['delta_temprature_4'] = train.apply(lambda df: minus_temperature(df['A25'], df['A27']), axis=1)
    train['delta_temprature_5'] = train.apply(lambda df: minus_temperature(df['A27'], df['B6']), axis=1)

    # 删除无用的温度特征
    train.drop(['A15', 'A17', 'A25'], axis=1, inplace=True)

    # 添加物质的量特征
    train['n_B1_B2'] = train.apply(lambda df: df['B1'] * df['B2'], axis=1)
    train['n_B13_B14'] = train.apply(lambda df: df['B13'] * df['B14'], axis=1)

    # 添加滴加速率
    train['v_B9'] = train.apply(lambda df: df['B12'] / df['B9'], axis=1) # 体积/时长=速率
    train['v_B10'] = train.apply(lambda df: df['B12'] / df['B10'], axis=1)
    train['v_B11'] = train.apply(lambda df: df['B12'] / df['B11'], axis=1)

    # 添加样本id特征
    train['id'] = train.apply(lambda df: id_features(df['样本id']), axis=1)
    train.drop(['样本id'], axis=1, inplace=True)

    return train


def FeaturesUnion(train, test, feature_categories):
    '''
    添加新特征，将收率进行分箱，然后构造每个特征中的类别对应不同收率的均值
    '''
    train['intTarget'] = pd.cut(train['收率'], 5, labels=False)
    train = pd.get_dummies(train, columns=['intTarget'])
    li = ['intTarget_' + str(idx) for idx in range(5)]
    mean_features = []

    target = train['收率']
    train.drop(['收率'], axis=1, inplace=True)

    # for data in [train, test]:
    #     for f in feature_categories:
    #         data[f] = data[f].map(dict(zip(data[f].unique(), range(0, data[f].nunique()))))

    for f1 in feature_categories:
        for f2 in li:
            col_name = f1 + "_" + f2 + '_mean'
            mean_features.append(col_name)
            order_label = train.groupby([f1])[f2].mean()
            # train[col_name] = train[f1].map(order_label)
            for df in [train, test]:
                df[col_name] = df[f1].map(order_label)

    # 保持train和test特征数目一致
    train.drop(li, axis=1, inplace=True)

    return train, test, target, mean_features


if __name__ == '__main__':
    # 读取数据
    train = pd.read_csv('data_transformed/jinnan_round1_train_20181227.csv', encoding = 'gb18030')
    test = pd.read_csv('data_transformed/jinnan_round1_testA_20181227.csv', encoding = 'gb18030')
    # Feature Engineering
    train = FeatureEngineering(train)
    test = FeatureEngineering(test)
    feature_categories = [name for name in train.columns if name != '收率' and name != 'id']
    train, test, target, mean_features = FeaturesUnion(train, test, feature_categories)

    # 保存
    # train.to_csv('data_transformed/features.csv', index=False, encoding='gb18030')


    # 分离变量
    X_train = train.values
    y_train = target.values
    X_test = test.values

    '''
    训练模型
    '''

    # # LGBoost
    param = {'num_leaves': 120,
             'min_data_in_leaf': 30,
             'objective': 'regression',
             'max_depth': -1,
             'learning_rate': 0.01,
             "min_child_samples": 30,
             "boosting": "gbdt",
             "feature_fraction": 0.9,
             "bagging_freq": 1,
             "bagging_fraction": 0.9,
             "bagging_seed": 11,
             "metric": 'mse',
             "lambda_l1": 0.1,
             "verbosity": -1}
    folds = KFold(n_splits=5, shuffle=True, random_state=2018)
    oof_lgb = np.zeros(len(train))
    predictions_lgb = np.zeros(len(test))
    
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
        print("fold n°{}".format(fold_ + 1))
        trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
        val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])
    
        num_round = 10000
        clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=200,
                        early_stopping_rounds=100)
        oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)
    
        predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits
    
    print("CV score: {:<8.8f}".format(mean_squared_error(oof_lgb, target)))

    # # XGBoost
    # ##### xgb
    xgb_params = {'eta': 0.005, 'max_depth': 10, 'subsample': 0.8, 'colsample_bytree': 0.8,
                  'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True, 'nthread': 4}

    folds = KFold(n_splits=5, shuffle=True, random_state=2018)
    oof_xgb = np.zeros(len(train))
    predictions_xgb = np.zeros(len(test))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
        print("fold n°{}".format(fold_ + 1))
        trn_data = xgb.DMatrix(X_train[trn_idx], y_train[trn_idx])
        val_data = xgb.DMatrix(X_train[val_idx], y_train[val_idx])

        watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
        clf = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200,
                        verbose_eval=100, params=xgb_params)
        oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]), ntree_limit=clf.best_ntree_limit)
        predictions_xgb += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits

    print("CV score: {:<8.8f}".format(mean_squared_error(oof_xgb, target)))

    # # 将lgb和xgb的结果进行stacking
    train_stack = np.vstack([oof_lgb, oof_xgb]).transpose()
    test_stack = np.vstack([predictions_lgb, predictions_xgb]).transpose()
    
    folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4590)
    oof_stack = np.zeros(train_stack.shape[0])
    predictions = np.zeros(test_stack.shape[0])
    
    for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack, target)):
        print("fold {}".format(fold_))
        trn_data, trn_y = train_stack[trn_idx], target.iloc[trn_idx].values
        val_data, val_y = train_stack[val_idx], target.iloc[val_idx].values
    
        clf_3 = BayesianRidge()
        clf_3.fit(trn_data, trn_y)
    
        oof_stack[val_idx] = clf_3.predict(val_data)
        predictions += clf_3.predict(test_stack) / 10
    
    mean_squared_error(target.values, oof_stack)

    '''
    输出预测结果
    '''
    sub_df = pd.read_csv('data_transformed/jinnan_round1_submit_20181227.csv', header=None)
    sub_df[1] = predictions
    sub_df[1] = sub_df[1].apply(lambda x:round(x, 3))

    sub_df.to_csv('data_transformed/result.csv', header=False, index=False)
