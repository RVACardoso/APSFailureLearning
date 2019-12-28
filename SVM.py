# if linearly separable -> use linear
# if non linearly separable -> use rbf

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dummy_var import preprocessData
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import ADASYN, SMOTE # http://glemaitre.github.io/imbalanced-learn/index.html
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

df_train = pd.read_csv('df_mean_train.csv')
df_test = pd.read_csv('df_mean_test.csv')

X_train_imb = df_train.iloc[:, 1:].values
y_train_imb = df_train.iloc[:, 0].values
X_test = df_test.iloc[:, 1:].values
y_test = df_test.iloc[:, 0].values

over_smote = SMOTE(random_state=57, ratio='auto')

# for over_met in [over_random, over_smote, over_adasyn]:
X_train_full, y_train_full = over_smote.fit_sample(X_train_imb, y_train_imb)
# must shuffle data after balancing
p = np.random.permutation(len(y_train_full))
X_train_full = X_train_full[p]
y_train_full = y_train_full[p]
# check class distribution
#print((y_train_full == 1).sum())
#print((y_train_full == 0).sum())

    # dont forget to scale data for svm:

# Create an object to transform the data to fit minmax processor
min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(X_train_full)
X_train_full = min_max_scaler.transform(X_train_full)
X_test = min_max_scaler.transform(X_test)

  ##  SVM
sys.stdout = open('svm_cv.txt', 'a')

print("\n SVM \n")

print('\n \n Grid Search on train set\n')

parameters_nonlin = {'C': [1, 10, 50, 100], 'gamma':'scale'}
classf = SVC() # choose here the best paramters determined via gridsearchCV

parameters_lin =

clf = GridSearchCV(classf, parameters, cv=10, n_jobs=-1, verbose=1)
clf.fit(X_train_full, y_train_full)
print("best score (mean cv): {}".format(clf.best_score_))
print("best parameters: {}".format(clf.best_params_))
print("mean test score ".format(clf.cv_results_['mean_test_score']))
print("std test score ".format(clf.cv_results_['std_test_score']))
df_results = pd.DataFrame(clf.cv_results_)
df_results.to_csv("df_knn_cv_results.csv", index=False)

# plt.figure()
# plt.scatter(list(range(1,10)), clf.cv_results_['mean_test_score'])
# plt.xlabel("Number of neighbors")
# plt.ylabel("Accuracy")
# plt.title("APS failure - Accuracy vs number of neighbors")


print('\n \n Test set \n')
y_pred_test = clf.predict(X_test)
y_true_test = y_test
print("Accuracy on test set: " + str(accuracy_score(y_true_test, y_pred_test)))

cm_test = confusion_matrix(y_true_test, y_pred_test)
print("Confusion matrix - test set")
print(cm_test)
# plt.figure()
# plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
# classNames = ['No', 'Yes']
# plt.title('Bank KNN Confusion Matrix - Train Data')
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# tick_marks = np.arange(len(classNames))
# plt.xticks(tick_marks, classNames, rotation=45)
# plt.yticks(tick_marks, classNames)
# for i in range(2):
#     for j in range(2):
#         plt.text(j,i, str(cm[i][j]))

TPrate = cm_test[1,1]/(cm_test[1,1]+cm_test[1,0]) # TPrate= TP/(TP+FN)
print("TPrate NB (test) = " + str(TPrate))
specif = cm_test[0,0]/(cm_test[0,0]+cm_test[0,1]) # specificity = TN/(TN+FP)
print("specificity NB (test) = " + str(specif))
cost = cm_test[0,1]*10 + cm_test[1,0]*500
print("cost (test) = " + str(cost))

# Compute ROC curve and ROC area for each class
y_score_test = clf.predict_proba(X_test)
y_score_test = np.array(y_score_test)[:,1]
fpr_test, tpr_test, _ = roc_curve(y_true_test, y_score_test)
roc_auc_test = auc(fpr_test, tpr_test)
print("auc (test) = " + str(roc_auc_test))


# plt.figure()
# plt.plot(fpr_nb, tpr_nb, color='darkorange', lw=2, label='NB curve (area = %0.2f)' % roc_auc_nb)
# plt.plot(fpr_knn, tpr_knn, color='darkgreen', lw=2, label='KNN curve (area = %0.2f)' % roc_auc_knn)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()

#df_read = pd.read_csv('df_knn_cv_results.csv')
#print(df_read.to_string())