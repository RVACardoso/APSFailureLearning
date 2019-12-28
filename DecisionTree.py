import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import preprocessing
import matplotlib.pyplot as plt
#import graphviz
import sys
# from dummy_var import preprocessData
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import ADASYN, SMOTE,  RandomOverSampler # http://glemaitre.github.io/imbalanced-learn/index.html
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from funcs_aps import k_fold_eval, true_pos_rate, specif, cost

# def to_nr(stri):
#     if stri == 'pos':
#         return 1
#     else:
#         return 0

# df_train = pd.read_csv('prpr_data/df_train_mean.csv')
# df_test = pd.read_csv('prpr_data/df_test_mean.csv')
#
# df_train = df_train.drop(columns=['cd_000'])
# df_test = df_test.drop(columns=['cd_000'])

df_train = pd.read_csv('prpr_data/df_badrm_train.csv')
df_test = pd.read_csv('prpr_data/df_badrm_test.csv')


X_train_imb = df_train.iloc[:, 1:].values
y_train_imb = df_train.iloc[:, 0].values
X_test = df_test.iloc[:, 1:].values
y_test = df_test.iloc[:, 0].values

ratio = {0: 59000, 1: 59000}
over_rand = RandomOverSampler(random_state=57, ratio=ratio)


#y_train_imb = [to_nr(elem) for elem in y_train_imb]  #for decision tree data

X_train_bal, y_train_bal = over_rand.fit_sample(X_train_imb, y_train_imb)
# must shuffle data after balancing
p = np.random.permutation(len(y_train_bal))
X_train_bal = X_train_bal[p]
y_train_bal = y_train_bal[p]
# check class distribution
#print((y_train_full == 1).sum())
# print((y_train_full == 0).sum())

##  Decision tree
print("starting tree...")

#sys.stdout = open('tree_cv.txt', 'a')

print("\n Decision tree \n")

#print('Preprocesing tuning \n')
#cart -> gini (default)
#c4.5 -> entri

#classf = tree.DecisionTreeClassifier()
# classf = tree.DecisionTreeClassifier(criterion='entropy')
#classf = tree.DecisionTreeClassifier(criterion='gini')

classf = tree.DecisionTreeClassifier(criterion='entropy', max_features=85, min_samples_split=2)
print(classf)
k_fold_eval(classf, X_train_bal, y_train_bal)

# clf.fit(X_train_full, y_train_full)


# print('\n \n Grid Search on train set STUMP\n')
#
# classf = tree.DecisionTreeClassifier(criterion='entropy') # choose here the best paramters determined via gridsearchCV
# parameters = {'min_samples_split': [2, 50, 100, 1000], 'max_features': [5, 13, 85, 170], 'max_depth':[1]}
# #parameters = {'min_samples_split': [2], 'max_features': [5]}
#
# clf = GridSearchCV(classf, parameters, cv=10, n_jobs=-1, verbose=1, refit=False,
#                    scoring = {'accur': 'accuracy', 'TPrate': true_pos_rate, 'specif': specif, 'cost': cost})
# clf.fit(X_train_bal, y_train_bal)
# print("\n \n \n results \n ")
# df_results = pd.DataFrame(clf.cv_results_)
# df_results.to_csv("df_tree_STUMP_grid_results.csv", index=False)


# STUMP
# classf = tree.DecisionTreeClassifier(criterion='entropy', max_features=85, min_samples_split=2, max_depth=1)
# graph = classf.fit(X_train_bal, y_train_bal)
# dot_data = tree.export_graphviz(graph, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("stump")
#
# k_fold_eval(classf, X_train_bal, y_train_bal)
#
#
# plt.show()
#


# print('\n \n Test set \n')
#
# classf = tree.DecisionTreeClassifier(criterion='entropy', max_features=85, min_samples_split=2)
# classf.fit(X_train_bal, y_train_bal)
# y_pred_test = classf.predict(X_test)
# y_true_test = y_test
# print("Accuracy on test set: " + str(accuracy_score(y_true_test, y_pred_test)))
#
# cm_test = confusion_matrix(y_true_test, y_pred_test)
# print("Confusion matrix - test set")
# print(cm_test)
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
# fpr_test, tpr_test, _ = roc_curve(y_true_test, y_score_test)
# roc_auc_test = auc(fpr_test, tpr_test)
# print("auc (test) = " + str(roc_auc_test))

# Decision stumps
# clf_c45 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=1)
# clf_c45_graph = clf_c45.fit(X_train_bal, y_train_bal)
#
# print("\n decision stumps")
#
# scores_c45 = cross_val_score(clf_c45, X, y, cv=10)
# scores_c45 = np.array(scores_c45)
# mean_acc_c45 = scores_c45.mean()
# print("Mean accuracy C4.5= " + str(mean_acc_c45))
# print("Std deviation C4.5= " + str(scores_c45.std()))
#
# scores_cart = cross_val_score(clf_cart, X, y, cv=10)
# scores_cart = np.array(scores_cart)
# mean_acc_cart = scores_cart.mean()
# print("Mean accuracy CART= " + str(mean_acc_cart))
# print("Std deviation CART= " + str(scores_cart.std()))

