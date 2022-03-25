import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import scikitplot as skplt
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import RandomOverSampler

def model_evaluation(Y_test,Y_pred):
	print(classification_report(Y_test,Y_pred))
	print("Accuracy Score: ",accuracy_score(Y_test,Y_pred))
	skplt.metrics.plot_confusion_matrix(Y_test, Y_pred)
	plt.show()
	



df=pd.read_csv("creditcard.csv")
data_features = df.iloc[:, 0:30]
data_targets = df.iloc[:, 30:]

oversample = RandomOverSampler(sampling_strategy='minority')
X_over, y_over = oversample.fit_resample(data_features, data_targets)

np.random.seed(42)
X_train, X_test, Y_train, Y_test = train_test_split(X_over, y_over,train_size = 0.70, test_size = 0.30, random_state = 1)

scaler1 = RobustScaler().fit(X_train['Amount'].values.reshape(-1,1))
scaler2 = RobustScaler().fit(X_train['Time'].values.reshape(-1,1))

X_train['scaled_amount'] = scaler1.transform(X_train['Amount'].values.reshape(-1,1))
X_train['scaled_time'] = scaler2.transform(X_train['Time'].values.reshape(-1,1))
X_train_scaled = X_train.drop(['Time','Amount'],axis = 1,inplace=False)

X_test['scaled_amount'] = scaler1.transform(X_test['Amount'].values.reshape(-1,1))
X_test['scaled_time'] = scaler2.transform(X_test['Time'].values.reshape(-1,1))
X_test_scaled = X_test.drop(['Time','Amount'],axis = 1,inplace=False)

lr = LogisticRegression(penalty="l2", C=5)
lr.fit(X_train_scaled,Y_train.values.ravel())
Y_pred=lr.predict(X_test_scaled)
model_evaluation(Y_test,Y_pred)
