#   The APS failure data is highly unbalanced and therefore requires application of data balancing algorithms
#   Lots of missing values (Imputing does not always improve the predictions, so please check via cross-validation. Sometimes dropping rows or using marker values is more effective)
#   usar decision tree ou outro algoritmo para preencher missing values


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
# http://glemaitre.github.io/imbalanced-learn/index.html
# https://medium.com/coinmonks/smote-and-adasyn-handling-imbalanced-data-set-34f5223e167
np.set_printoptions(threshold=np.nan)

def create_df(df_train, X_pos, X_neg, out_file, create_file=None, random=None):
    X_test = np.vstack((X_pos, X_neg))
    y_test = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))  # 1 for pos, 0 for neg
    data_mean = np.column_stack((y_test, X_test))
    df_mean = pd.DataFrame(data_mean)
    df_mean.columns = df_train.keys()
    if random == 1:
        df_mean = df_mean.sample(frac=1)
    #print("pos count: {}".format((df_mean.iloc[:, 0].values == 1).sum()))
    #print("neg count: {}".format((df_mean.iloc[:, 0].values == 0).sum()))
    if create_file == 1:
        df_mean.to_csv(out_file, index=False)
    return df_mean


def impute_mean(df_train, df_test=None, create_file=None):
    X_train_pos = df_train.loc[df_train['class'] == 'pos'].iloc[:, 1:].values
    X_train_neg = df_train.loc[df_train['class'] == 'neg'].iloc[:, 1:].values
    #print("Shape of X_train pos: " + str(X_train_pos.shape))
    #print("Shape of X_train neg: " + str(X_train_neg.shape))
    imput_pos = SimpleImputer(strategy="mean")
    X_train_pos = imput_pos.fit_transform(X_train_pos)
    imput_neg = SimpleImputer(strategy="mean")
    X_train_neg = imput_neg.fit_transform(X_train_neg)
    df_train_mean = create_df(df_train, X_train_pos, X_train_neg, out_file="df_train_mean.csv", create_file=create_file)

    if df_test is not None:
        X_test_pos = df_test.loc[df_test['class'] == 'pos'].iloc[:, 1:].values
        X_test_neg = df_test.loc[df_test['class'] == 'neg'].iloc[:, 1:].values
        #print("Shape of X_test pos: " + str(X_test_pos.shape))
        #print("Shape of X_test neg: " + str(X_test_neg.shape))
        X_test_pos = imput_pos.transform(X_test_pos)
        X_test_neg = imput_neg.transform(X_test_neg)
        df_test_mean = create_df(df_test, X_test_pos, X_test_neg, out_file="df_test_mean.csv", create_file=create_file)
        return df_train_mean, df_test_mean
    else:
        return df_train_mean




train_datafr = pd.read_csv("data/aps_failure_training_set.csv", na_values='na', keep_default_na=True, na_filter=True, skiprows=range(20))
test_datafr = pd.read_csv("data/aps_failure_test_set.csv", na_values='na', keep_default_na=True, na_filter=True, skiprows=range(20))

print(train_datafr.head(20).to_string())

#print(train_datafr.isnull().sum(axis=0).to_string())# > 0).sum())
#print((train_datafr.isnull().sum(axis=1) == 0).sum())


#   Handling missing values

    # Fill in with mean of samples belonging to the same class

train_mean_imp, test_mean_imp = impute_mean(df_train=train_datafr, df_test=test_datafr, create_file=1)
#print(train_mean_imp.iloc[:50, :].to_string())

    # Fill in with decision tree
#
# for feat in train_datafr.keys()[1:]:
#     print(feat)
#     nan_rows = train_datafr[feat].index[train_datafr[feat].apply(np.isnan)]
#     nan_rows_test = test_datafr[feat].index[test_datafr[feat].apply(np.isnan)]
#     if len(nan_rows) == 0:
#         continue
#     non_nan_rows = train_datafr[feat].index[~train_datafr[feat].apply(np.isnan)]
#
#     train_feat_in = train_datafr.iloc[non_nan_rows, :]
#     pred_feat = train_datafr.iloc[nan_rows, :]
#     pred_feat_test = test_datafr.iloc[nan_rows_test, :]
#
#     train_feat, pred_feat = impute_mean(df_train=train_feat_in, df_test=pred_feat)
#     train_feat_pos = train_feat.loc[train_feat['class'] == 1].iloc[:, 1:]
#     train_feat_neg = train_feat.loc[train_feat['class'] == 0].iloc[:, 1:]
#     _, pred_feat_test = impute_mean(df_train=train_feat_in, df_test=pred_feat_test)
#
#     tree_pos = DecisionTreeRegressor(random_state=0)
#     tree_pos.fit(train_feat_pos.drop(feat, axis=1).values, train_feat_pos[feat].values)
#     tree_neg = DecisionTreeRegressor(random_state=0)
#     tree_neg.fit(train_feat_neg.drop(feat, axis=1).values, train_feat_neg[feat].values)
#
#     pred_feat_pos = pred_feat.loc[pred_feat['class'] == 1].iloc[:, 1:]
#     pred_feat_neg = pred_feat.loc[pred_feat['class'] == 0].iloc[:, 1:]
#     pred_feat_test_pos = pred_feat_test.loc[pred_feat_test['class'] == 1].iloc[:, 1:]
#     pred_feat_test_neg = pred_feat_test.loc[pred_feat_test['class'] == 0].iloc[:, 1:]
#
#
#     imput_pos = tree_pos.predict(pred_feat_pos.loc[:, pred_feat_pos.columns != feat].values)
#     imput_neg = tree_neg.predict(pred_feat_neg.loc[:, pred_feat_neg.columns != feat].values)
#
#     nan_pos_rows = train_datafr.iloc[nan_rows, :].loc[train_datafr['class'] == 'pos'].index
#     nan_neg_rows = train_datafr.iloc[nan_rows, :].loc[train_datafr['class'] == 'neg'].index
#
#     train_datafr[feat].iloc[nan_pos_rows] = imput_pos
#     train_datafr[feat].iloc[nan_neg_rows] = imput_neg
#
#     if pred_feat_test_pos.shape[0] != 0:
#         imput_test_pos = tree_pos.predict(pred_feat_test_pos.loc[:, pred_feat_test_pos.columns != feat].values)
#         nan_pos_rows_test = test_datafr.iloc[nan_rows_test, :].loc[test_datafr['class'] == 'pos'].index
#         test_datafr[feat].iloc[nan_pos_rows_test] = imput_test_pos
#
#     if pred_feat_test_neg.shape[0] != 0:
#         imput_test_neg = tree_neg.predict(pred_feat_test_neg.loc[:, pred_feat_test_neg.columns != feat].values)
#         nan_neg_rows_test = test_datafr.iloc[nan_rows_test, :].loc[test_datafr['class'] == 'neg'].index
#         test_datafr[feat].iloc[nan_neg_rows_test] = imput_test_neg
#
# train_datafr.to_csv("./prpr_data/df_train_tree.csv", index=False)
# test_datafr.to_csv("./prpr_data/df_test_tree.csv", index=False)

