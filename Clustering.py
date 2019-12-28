import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
# import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contg_matrix = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contg_matrix, axis=0)) / np.sum(contg_matrix)


def cluster(clf, X_plt, y_plt, title):
    result = clf.fit_predict(X_plt)
    print(np.unique(result))
    silh_scr = silhouette_score(X_plt, result)
    print('\n silh score for ' + title + ': ')
    rand_ind = adjusted_rand_score(y_plt, result)
    print('adjusted rand index: ')
    purity = purity_score(y_plt, result)
    print('purity: ')
    print(str(round(silh_scr, 5)))
    print(str(round(rand_ind, 5)))
    print(str(round(purity, 5)))



df_train = pd.read_csv('prpr_data/df_train_mean.csv')
df_train = df_train.drop(columns=['cd_000'])

# stand_list = []
# for col in list(df_train):
#     stand = np.std(df_train[col].values)
#     stand_list.append(stand)
# print(stand_list)

X_train_imb = df_train.iloc[:, 1:].values
y_train_imb = df_train.iloc[:, 0].values

# min_max_scaler = preprocessing.MinMaxScaler()
# min_max_scaler.fit(X_train_imb)
# X_train_imb = min_max_scaler.transform(X_train_imb)

# classf = AgglomerativeClustering(n_clusters=2, linkage='single')
# cluster(classf, X_train_imb, y_train_imb, title='aglom single')
# classf = AgglomerativeClustering(n_clusters=2, linkage='average')
# cluster(classf, X_train_imb, y_train_imb, title='aglom average')
# classf = AgglomerativeClustering(n_clusters=2, linkage='complete')
# cluster(classf, X_train_imb, y_train_imb, title='aglom complete')
# classf = AgglomerativeClustering(n_clusters=2, linkage='ward')
# cluster(classf, X_train_imb, y_train_imb, title='aglom ward')

# for i in [7,8,9,10]:
#     print('\n '+ str(i))
#     kmeans = KMeans(n_clusters=i, random_state=0, n_jobs=-1, verbose=0)
#     cluster(kmeans, X_train_imb, y_train_imb, title='k-mean')
# #

# for eps_val in [2000, 5000, 7500]:
#     try:
#         dbscan = DBSCAN(eps=eps_val, min_samples=11, n_jobs=-1) #min_sample = ln 60000
#         cluster(dbscan, X_train_imb, y_train_imb, title='dbscan')
#         print("made it " + str(eps_val))
#     except:
#         print("nope " + str(eps_val))

#EXPECTATION MAXIMIZATION ALGORIThm
gauss = GaussianMixture(n_components=9, init_params='kmeans', reg_covar=1e1)
cluster(gauss, X_train_imb, y_train_imb, title='gauss')


# CHECK THIS: http://scikit-learn.org/stable/modules/clustering.html

#-0.08705433468170272