from sklearn.manifold import TSNE
from sklearn.datasets import load_iris, load_diabetes, load_digits
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets.samples_generator import make_classification
import warnings

warnings.filterwarnings('ignore')


def model(classifier_name, X, y):
	print(classifier_name)
	print('data shape:{0}'.format(X.shape))
	classifier = RandomForestClassifier()
	X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	# print('X_train shape:{0}'.format(X_train.shape))
	# print('x_test shape:{0}'.format(x_test.shape))
	# print('y_train shape:{0}'.format(y_train.shape))
	# print('y_test shape:{0}'.format(y_test.shape))
	
	classifier.fit(X=X_train, y=y_train)
	y_predict = classifier.predict(X=x_test)
	y_predict_proba = classifier.predict_proba(x_test)
	precision = metrics.precision_score(y_test, y_pred=y_predict, average='macro')
	print('precision:{0}'.format(precision))


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


def normalize(data):
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





def main():
	# load data
	n_class = 5
	train_data, train_label, n_samples, n_features = load_data(n_class=n_class)
	train_data = normalize(train_data)
	
	
	
	#Evaluate performance of model using original data
	model('original data',train_data,train_label)
	
	
	
	
	#TSNE,evaluate performance of model using features which be extracted by TSNE
	tsne = TSNE(n_components=3,random_state=42,perplexity=20,n_iter=3000)
	result1 = tsne.fit_transform(X=train_data)
	model('TSNE',result1,train_label)
	_plot(X=result1,y=train_label,labels=None,title='TSNE')
	
	
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
	

main()
