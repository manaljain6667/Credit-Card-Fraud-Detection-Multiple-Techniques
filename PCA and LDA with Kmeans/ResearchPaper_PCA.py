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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def model_evaluation(Y_test,Y_pred):
	print(classification_report(Y_test,Y_pred))
	print("Accuracy Score: ",accuracy_score(Y_test,Y_pred))
	skplt.metrics.plot_confusion_matrix(Y_test, Y_pred)
	plt.show()
	

df=pd.read_csv("creditcard.csv")
data_features = df.iloc[:, 1:30]
data_targets = df.iloc[:, 30:]

oversample = RandomOverSampler(sampling_strategy='minority')
X_over, y_over = oversample.fit_resample(data_features, data_targets)


np.random.seed(42)
X_train, X_test, Y_train, Y_test = train_test_split(X_over, y_over,train_size = 0.70, test_size = 0.30, random_state = 1)

scaler1 = RobustScaler().fit(X_train['Amount'].values.reshape(-1,1))


X_train['scaled_amount'] = scaler1.transform(X_train['Amount'].values.reshape(-1,1))
X_train_scaled = X_train.drop(['Amount'],axis = 1,inplace=False)
X_test['scaled_amount'] = scaler1.transform(X_test['Amount'].values.reshape(-1,1))
X_test_scaled = X_test.drop(['Amount'],axis = 1,inplace=False)


principal=PCA(n_components=5)
principal.fit(X_train_scaled)
X_train_pca=principal.transform(X_train_scaled)
X_test_pca=principal.transform(X_test_scaled)

print(X_train_pca.shape)
#pd.DataFrame(x).to_csv("file.csv")




#kmeans = KMeans(n_clusters=2, random_state=0, algorithm="full", max_iter=1000)
kmeans = KMeans(n_clusters=2, init='random',n_init=10, max_iter=1000,tol=1e-04, random_state=0)
kmeans.fit(X_train_pca)
kmeans_predicted_cluster_for_train=kmeans.predict(X_train_pca)

print(pd.DataFrame(kmeans_predicted_cluster_for_train).nunique())
print(kmeans_predicted_cluster_for_train.shape)
tn, fp, fn, tp = confusion_matrix(Y_train,kmeans_predicted_cluster_for_train).ravel()
reversed_cluster=False
if tn+tp<fn+fp:
	reversed_cluster=True
kmeans_predicted_cluster_for_test = kmeans.predict(X_test_pca)
if reversed_cluster:
	kmeans_predicted_cluster_for_test = 1 - kmeans_predicted_cluster_for_test
model_evaluation(Y_test,kmeans_predicted_cluster_for_test)