from sklearn.manifold import TSNE
from sklearn.datasets import load_iris, load_diabetes, load_digits
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
# from sklearn.datasets.samples_generator import make_classification
from collections import Counter
import warnings

warnings.filterwarnings('ignore')


def model(classifier_name, X, y):
	print(classifier_name)
	print('data shape:{0}'.format(X.shape))
	# classifier = RandomForestClassifier(n_estimators=400)
	classifier = xgb.XGBClassifier(silent=False,
                      scale_pos_weight=1,
                      learning_rate=0.01,
                      colsample_bytree = 0.4,
                      subsample = 0.5,
                      objective='binary:logistic',
                      n_estimators=1000,
                      reg_alpha = 0.5,
                      max_depth=6,
                      gamma=10)
	X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	# print('X_train shape:{0}'.format(X_train.shape))
	# print('x_test shape:{0}'.format(x_test.shape))
	# print('y_train shape:{0}'.format(y_train.shape))
	# print('y_test shape:{0}'.format(y_test.shape))
	
	classifier.fit(X_train, y_train)
	y_predict = classifier.predict(x_test)
	y_predict_proba = classifier.predict_proba(x_test)
	# metrics.plot_roc_curve(estimator=classifier,X=x_test,y=y_test)
	precision = metrics.precision_score(y_test, y_pred=y_predict, average='macro')
	accuracy = metrics.accuracy_score(y_true=y_test,y_pred=y_predict)
	print('precision:{0}'.format(precision))
	print('accracy:{0}'.format(accuracy))


def load_data(n_class):
	data = load_iris()
	data = load_diabetes()
	data = load_digits(n_class=n_class)
	train_data = data['data']
	train_label = data['target']
	print(train_label[:10])
	n_samples = train_data.shape[0]
	n_features = train_data.shape[1]
	return train_data, train_label, n_samples, n_features


def load_digits_data():
	df = pd.read_csv('digit-recognizer/train.csv',sep=',',index_col=None)
	df = df.sample(frac=1)
	y = df['label']
	columns = df.columns.tolist()
	X = df.loc[:,columns[1:]]
	X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.1,random_state=42)
	print('X_train shape:{0}'.format(X_train.shape))
	print('X_val shape:{0}'.format(X_val.shape))
	print('y_train shape:{0}'.format(y_train.shape))
	print('y_val shape:{0}'.format(y_val.shape))

	return 	X_train, X_val, y_train, y_val

def load_test():
	df = pd.read_csv('digit-recognizer/test.csv',sep=',',index_col = None)
	print(df.shape)
	print(df.columns)
	return df.to_numpy()

def mynormalize(data):
	"""
	Rescaling data to have values between 0 and 1
	:param data:numpy array ,shape:(n_samples,n_features)
	:return:
	"""
	data = (data - np.min(data)) / (np.max(data) - np.min(data))
	return data


def _plot(X, y, labels, title):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker='o', c=y)
	ax.set_xlabel('X label')
	ax.set_ylabel('Y label')
	ax.set_zlabel('Z label')
	ax.set_title(title)
	plt.show()


def evaluate_performance(prefix, classifier, X, y):
	"""
	:param classifier: Trained classifier
	:param X: array-like of shape (n_samples,n_features)
	:param y:array-like of shape (n_samples)
	:return: None
	"""
	print(prefix)
	y_predict = classifier.predict(X)
	print('y_val counter:{0}'.format(Counter(y)))
	print('y_predict counter:{0}'.format(Counter(y_predict)))
	y_predict_proba = classifier.predict_proba(X)
	precision = metrics.precision_score(y, y_pred=y_predict, average='macro')
	recall = metrics.recall_score(y, y_predict,average='macro')
	accuracy = metrics.accuracy_score(y_true=y,y_pred=y_predict)
	print('precision:{0}'.format(precision))
	print('recall:{0}'.format(recall))
	print('accuracy:{0}'.format(accuracy))
	# print('roc_auc_score:{0}'.format(metrics.roc_auc_score(y_true=y, y_score=y_predict,multi_class='ovo')))
	# metrics.plot_roc_curve(estimator=classifier, X=X, y=y)
	# metrics.plot_confusion_matrix(classifier, X, y)

def model_v1(classifier_name, X, y):
	print(classifier_name)
	print('data shape:{0}'.format(X.shape))
	classifier = RandomForestClassifier(n_estimators=200,n_jobs=26)
	# classifier = xgb.XGBClassifier(silent=False,
    #                   scale_pos_weight=1,
    #                   learning_rate=0.01,
    #                   colsample_bytree = 0.4,
    #                   subsample = 0.5,
    #                   objective='binary:logistic',
    #                   n_estimators=1000,
    #                   reg_alpha = 0.5,
    #                   max_depth=6,
    #                   gamma=10,
    #                   n_jobs=26)
	# classifier = tree.DecisionTreeClassifier()
	classifier.fit(X=X, y=y)
	# jl.dump(classifier,'{0}.joblib'.format(classifier_name))
	return classifier

def predict(classifier,X):
	y_predict = classifier.predict(X)
	return y_predict


def main():
	# load data
	# n_class = 10
	# train_data, train_label, n_samples, n_features = load_data(n_class=n_class)
	# train_data = mynormalize(train_data)
	
	
	
	#Evaluate performance of model using original data
	# model('baseline',train_data,train_label)

	
	
	
	#TSNE,evaluate performance of model using features which be extracted by TSNE
	# tsne = TSNE(n_components=3,random_state=42,perplexity=20,n_iter=3000)
	# result1 = tsne.fit_transform(X=train_data)
	# model('TSNE',result1,train_label)
	# _plot(X=result1,y=train_label,labels=None,title='TSNE')
	
	
	# #PCA,evaluate performance fo model using features which be extracted by PCA
	# pca = PCA(n_components=n_class, random_state=42)
	# result2 = pca.fit_transform(X=train_data)
	# model('PCA', result2, train_label)
	# _plot(X=result2,y=train_label,labels=None,title='PCA')
	#
	#
	# #LDA
	# lda = LinearDiscriminantAnalysis(n_components=n_class)
	# X_new = lda.fit_transform(X=train_data,y=train_label)
	# model('LDA',X_new,train_label)
	# _plot(X=X_new,y=train_label,labels=None,title='LDA')
	#
	#
	#
	# #KMeans,Evaluate performance of model using features which be extracted by KMeans
	# km = KMeans(n_clusters=n_class,random_state=42)
	# result = km.fit_transform(X=train_data)
	# model('kMeans',result,train_label)
	# _plot(X=result,y=train_label,labels=None,title='KMeans')
	
	
	
	#digits recognize
	X_train, X_val, y_train, y_val = load_digits_data()
	X_test = load_test()
	print('test shape:{0}'.format(X_test.shape))
	# tsne = TSNE(n_components=3,random_state=42,perplexity=20,n_iter=3000,n_jobs=10)
	# X_train = tsne.fit_transform(X=X_train)
	# X_val = tsne.fit_transform(X=X_train)
	# X_test = tsne.fit_transform(X=X_test)
	
	
	# pca = PCA(n_components=60,random_state=42)
	# X_train = pca.fit_transform(X_train,y_train)
	# X_val = pca.transform(X_val)
	# X_test = pca.transform(X_test)
	
	
	# kmeans = KMeans(n_clusters=80,n_jobs=20)
	# X_train = kmeans.fit_transform(X_train,y_train)
	# X_val = kmeans.transform(X_val)
	# X_test = kmeans.transform(X_test)
	
	
	classifier = model_v1('Digit Recognizer',X_train,y_train)
	evaluate_performance('Digit Recognizer',classifier,X_val,y_val)
	y_predict = predict(classifier,X_test)
	print(y_predict.shape)
	ImageId = np.arange(1,len(y_predict)+1)
	df = pd.DataFrame({'ImageId':ImageId,'Label':y_predict})
	df.to_csv('digit-recognizer/sample_submission.csv',sep=',',index=None)
	print(df.head())

main()
