#bernouli multinomial ou gaussiana


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from sklearn.model_selection import KFold
import sys
from funcs_aps import k_fold_eval

np.set_printoptions(threshold=np.nan)

#df_train = pd.read_csv('prpr_data/df_train_mean.csv')
#df_test = pd.read_csv('prpr_data/df_test_mean.csv')

df_train = pd.read_csv('prpr_data/df_badrm_train.csv')
df_test = pd.read_csv('prpr_data/df_badrm_test.csv')


# df_train = df_train.drop(columns=['cd_000'])
# df_test = df_test.drop(columns=['cd_000'])

# col = [17, 63, 68, 83, 95]
# col = [elem+15 for elem in col]
# print(col)
# X_train_imb = df_train.iloc[:, col].values

X_train_imb = df_train.iloc[:, 1:].values
y_train_imb = df_train.iloc[:, 0].values
X_test = df_test.iloc[:, 1:].values
y_test = df_test.iloc[:, 0].values

over_smote = SMOTE(random_state=57, ratio='auto')
X_train_bal, y_train_bal = over_smote.fit_sample(X_train_imb, y_train_imb)
p = np.random.permutation(len(y_train_bal))
X_train_bal = X_train_bal[p]
y_train_bal = y_train_bal[p]

# dont forget to scale data for naive bayes:
min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(X_train_bal)
X_train_bal = min_max_scaler.transform(X_train_bal)
#X_test = min_max_scaler.transform(X_test)

#sys.stdout = open('NBayes_cv.txt', 'a')

    ## NAIVE BAYES

#classf = GaussianNB()
classf = MultinomialNB()
#classf = ComplementNB()
# classf = BernoulliNB()
print("\n \n")
print(classf)

#classf.fit(X_train_bal, y_train_bal)

k_fold_eval(classf, X_train_bal, y_train_bal)

#     # for test set

# print('\n \n test set \n')
# y_pred_test = classf.predict(X_test)
# y_true_test = y_test
#
#
# print("Accuracy on test set: " + str(accuracy_score(y_true_test, y_pred_test)))
#
# cm_test = confusion_matrix(y_true_test, y_pred_test)
# print("Confusion matrix Test set")
# print(cm_test)
#
# TPrate = cm_test[1,1]/(cm_test[1,1]+cm_test[1,0]) # TPrate= TP/(TP+FN)
# print("TPrate NB (test) = " + str(TPrate))
# specif = cm_test[0,0]/(cm_test[0,0]+cm_test[0,1]) # specificity = TN/(TN+FP)
# print("specificity NB (test) = " + str(specif))
# cost = cm_test[0,1]*10 + cm_test[1,0]*500
# print("cost (test) = " + str(cost))
#
# # Compute ROC curve and ROC area for each class
# y_score_test = classf.predict_proba(X_test)
# y_score_test = np.array(y_score_test)[:,1]
# fpr_nb_test, tpr_nb_test, _ = roc_curve(y_true_test, y_score_test)
# roc_auc_nb_test = auc(fpr_nb_test, tpr_nb_test)
# print("auc (test) = " + str(roc_auc_nb_test))

