import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import preprocessing
import matplotlib.pyplot as plt
# import graphviz
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
from sklearn.ensemble import RandomForestClassifier
from funcs_aps import k_fold_eval, true_pos_rate, specif, cost

def to_nr(stri):
    if stri == 'pos':
        return 1
    else:
        return 0

df_train = pd.read_csv('prpr_data/df_train_mean.csv')
df_test = pd.read_csv('prpr_data/df_test_mean.csv')

df_train = df_train.drop(columns=['cd_000'])
df_test = df_test.drop(columns=['cd_000'])

X_train_imb = df_train.iloc[:, 1:].values
y_train_imb = df_train.iloc[:, 0].values
X_test = df_test.iloc[:, 1:].values
y_test = df_test.iloc[:, 0].values

ratio = {0: 59000, 1: 10000}
# over_smote = SMOTE(random_state=57, ratio=ratio)
over_rand = RandomOverSampler(random_state=57, ratio=ratio)


#y_train_imb = [to_nr(elem) for elem in y_train_imb]  #for decision tree data

X_train_bal, y_train_bal = over_rand.fit_sample(X_train_imb, y_train_imb)
# must shuffle data after balancing
p = np.random.permutation(len(y_train_bal))
X_train_bal = X_train_bal[p]
y_train_bal = y_train_bal[p]
# check class distribution
# print((y_train_full == 1).sum())
# print((y_train_full == 0).sum())

##  Random Forest
print("starting forest...")

#sys.stdout = open('tree_cv.txt', 'a')

print("\n Random Forest \n")

#print('Preprocesing tuning \n')

#classf = tree.DecisionTreeClassifier()
# classf = tree.DecisionTreeClassifier(criterion='entropy')
#classf = tree.DecisionTreeClassifier(criterion='gini')

# classf = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=33)
#
# print(classf)
# k_fold_eval(classf, X_train_bal, y_train_bal)


# print('\n \n Grid Search on train set \n')
#
#
# # usar melhores parametros escolhidos para as decision trees
#
# classf = RandomForestClassifier(criterion='entropy', random_state=33, max_features=85, min_samples_split=2)
# parameters = {'n_estimators': [10, 100, 200], 'max_depth': [1, 10, 50, 100]}
#
# clf = GridSearchCV(classf, parameters, cv=10, n_jobs=-1, verbose=5, refit=False,
#                    scoring = {'accur': 'accuracy', 'TPrate': true_pos_rate, 'specif': specif, 'cost': cost})
# clf.fit(X_train_bal, y_train_bal)
# print("\n \n \n results \n ")
# df_results = pd.DataFrame(clf.cv_results_)
# df_results.to_csv("df_forest_grid_results.csv", index=False)


print('\n \n Test set \n')

classf = RandomForestClassifier(criterion='entropy', max_features=85, min_samples_split=2, max_depth=50, n_estimators=10)
classf.fit(X_train_bal, y_train_bal)
y_pred_test = classf.predict(X_test)
y_true_test = y_test
print("Accuracy on test set: " + str(accuracy_score(y_true_test, y_pred_test)))

cm_test = confusion_matrix(y_true_test, y_pred_test)
print("Confusion matrix - test set")
print(cm_test)
TPrate = cm_test[1,1]/(cm_test[1,1]+cm_test[1,0]) # TPrate= TP/(TP+FN)
print("TPrate NB (test) = " + str(TPrate))
specif = cm_test[0,0]/(cm_test[0,0]+cm_test[0,1]) # specificity = TN/(TN+FP)
print("specificity NB (test) = " + str(specif))
cost = cm_test[0,1]*10 + cm_test[1,0]*500
print("cost (test) = " + str(cost))

# Compute ROC curve and ROC area for each class
y_score_test = classf.predict_proba(X_test)
y_score_test = np.array(y_score_test)[:,1]
fpr_test, tpr_test, _ = roc_curve(y_true_test, y_score_test)
roc_auc_test = auc(fpr_test, tpr_test)
print("auc (test) = " + str(roc_auc_test))


# forest of stumps???
