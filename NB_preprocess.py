df_train = pd.read_csv('prpr_data/df_train_mean.csv')
df_test = pd.read_csv('prpr_data/df_test_mean.csv')

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
X_test = min_max_scaler.transform(X_test)


