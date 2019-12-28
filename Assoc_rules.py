import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from mlxtend.frequent_patterns import apriori, association_rules


# maybe fazer class condiitonal association rules


def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) <= m * np.std(data)]

def unique_ordered(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def old_find_edges(data, bins=500): # receives data column
    histo, bin_edges = np.histogram(data, bins=bins)  # arguments are passed to np.histogram
    #print(histo)

    histo_non = histo[np.nonzero(histo)]
    #print(histo_non)
    order = unique_ordered(histo_non)
    order_len = len(order)
    #print(order[round(order_len/3.0)])
    #print(order[round(2*order_len/3.0)])

    itemindex = np.where(histo==order[round(order_len/3.0)])[0][0]
    #print(itemindex)
    #print(histo[itemindex])
    first_edge = bin_edges[itemindex]
    #print(first_edge)

    itemindex = np.where(histo==order[order_len-1])[0][0]
    #print(itemindex)
    #print(itemindex)
    #print(histo[itemindex])
    sec_edge = bin_edges[itemindex]
    #print(sec_edge)
    return first_edge, sec_edge

def find_edges(data, bins=500):
    mean = np.mean(data)
    std = np.std(data)
    min_val = np.min(data)-0.000001
    max_val = np.max(data)+0.000001
    edges = [min_val]
    for i in [-2, -1, 0, 1, 2]:
        value = mean + i*std
        if value < max_val and value > min_val:
            #edges.append(value)
            edges.append(value)
    edges.append(max_val)
    return np.array(edges)

def clean_items(item_in):
    item_out = [elem for elem in item_in]
    return item_out


#df_train = pd.read_csv('prpr_data/df_train_mean.csv')
df_train = pd.read_csv('prpr_data/df_badrm_train.csv')

# print(find_edges(df_train['ay_000']))
#
# df_test = pd.read_csv('prpr_data/df_test_mean.csv')
#
# X_train_imb = df_train.iloc[:, 73].values
# y_train_imb = df_train.iloc[:, 0].values
# X_test = df_test.iloc[:, 1:].values
# y_test = df_test.iloc[:, 0].values


# for col in range(1, 171):
#     print(col)
#     X_train_imb = df_train.iloc[:, col].values
#     data_col = reject_outliers(data=X_train_imb, m=20)
#     edges = find_edges(data_col)
#
#     plt.figure()
#     plt.hist(data_col, bins=500)  # arguments are passed to np.histogram
#     for i in edges:
#         plt.axvline(x=i, color='r')
#     plt.title('col nr: ' + str(col))
#     #plt.show()
#
#

#df_train = df_train.drop(columns=['cd_000'])
#
# for col in list(df_train.iloc[:, 1:]):
#     print(col)
#     edges = find_edges(df_train[col])
#     print(edges)
#     df_train[col] = pd.cut(df_train[col], bins=edges, labels=[nr for nr in range(np.unique(edges).shape[0]-1)], duplicates='drop')
#     attrs = []
#     values = np.unique(df_train[col].values)
#     values.sort()
#     for val in values: attrs.append("%s:%s"%(col,val))
#     lb = LabelBinarizer().fit_transform(df_train[col])
#     df2 = pd.DataFrame(data=lb, columns=attrs)
#     for col2 in list(df2):
#         if df2[col2].sum() == 0:
#             print("dropping")
#             df2=df2.drop(columns=[col2])
#     df_train = df_train.drop(columns=[col])
#     df_train = pd.concat([df_train, df2], axis=1, join='inner')
#
# #print(df_train)
#
# pos = df_train.loc[df_train['class'] == 1].head(150)
# neg = df_train.loc[df_train['class'] == 0].head(150)
# binar = pd.concat([pos, neg])
# print(binar.head(5).to_string())
# print(binar.shape)
#
#
# binar = binar.drop(columns=['class'])
# # print(df_train.head(5).to_string())
# binar.to_csv("binarized_train.csv", index=False)
# print("Binarization complete!")


df_binarized = pd.read_csv('binarized_train.csv')
print(df_binarized.shape)
for col2 in list(df_binarized):
    if df_binarized[col2].sum() == 0:
        df_binarized = df_binarized.drop(columns=[col2])

print(df_binarized.shape)

supp_ = []
rule_cnt_ = []
lift_ = []
#for min_supp in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
min_supp = 0.95

frequent_itemsets = apriori(df_binarized, min_support=min_supp, use_colnames=True, max_len=5)#97
frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(clean_items)

print("item shape " + str(frequent_itemsets.shape))

print('\n one more')
# rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)#02
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)#02
rules = rules.sort_values(by='lift', ascending=False)
#rules.head(50).to_csv("rules_95_2.csv", index=False)
lift_.append(np.mean(rules.head(20)['lift']))
print('all lifts: ' + str(np.mean(rules['lift'])))

#print("rules shape " + str(rules.shape))
supp_.append(min_supp)
rule_cnt_.append(rules.shape[0])

print(supp_)
print('\n')
print(rule_cnt_)
print('\n')
print(lift_)

# plt.plot(supp_, rule_cnt_)
# plt.show()





# rules = pd.read_csv("rules_95_treat.csv")
# print(rules.to_string())