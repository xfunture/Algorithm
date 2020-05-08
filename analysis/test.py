from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def load_data():
	data = load_iris()
	print(type(data['data']))
	train_data = data['data']
	train_label = data['target']
	n_samples = train_data.shape[0]
	n_features = train_data.shape[1]
	return train_data,train_label,n_samples,n_features

def normalize(data):
	"""
	Rescaling data to have values between 0 and 1
	:param data:numpy array ,shape:(n_samples,n_features)
	:return:
	"""
	data = (data - np.min(data))/(np.max(data) - np.min(data))
	return data

def main():
	train_data,train_label,n_samples,n_features = load_data()
	train_data = normalize(train_data)
	print(train_label.shape)
	
	tsne = TSNE(n_components=3,random_state=42,perplexity=20,n_iter=5000)
	result1 = tsne.fit_transform(X=train_data)
	print(result1.shape)
	# fig = plt.figure(figsize=(20,10))
	fig = plt.figure()
	ax1 = fig.add_subplot(111, projection='3d')
	# ax = Axes3D(fig)
	for i in range(result1.shape[0]):
		if train_label[i] == 0:
			ax1.scatter(result1[i,0],result1[i,1],result1[i,2],c='r')
		elif train_label[i] == 1:
			ax1.scatter(result1[i,0],result1[i,1],result1[i,2],c='b')
		else:
			ax1.scatter(result1[i,0],result1[i,1],result1[i,2],c='y')
	ax1.set_xlabel('X label')
	ax1.set_ylabel('Y label')
	ax1.set_zlabel('Z label')
	ax1.set_title('SNE')
	#
	# pca = PCA(n_components=3, random_state=42)
	# result2 = pca.fit_transform(X=train_data)
	# print(result2.shape)
	# ax2 = fig.add_subplot(212, projection='3d')
	# # ax = Axes3D(fig)
	# for i in range(result1.shape[0]):
	# 	if train_label[i] == 0:
	# 		ax2.scatter(result2[i, 0], result2[i, 1], result2[i, 2], c='r')
	# 	elif train_label[i] == 1:
	# 		ax2.scatter(result2[i, 0], result2[i, 1], result2[i, 2], c='b')
	# 	else:
	# 		ax2.scatter(result2[i, 0], result2[i, 1], result2[i, 2], c='y')
	# ax2.set_title('PCA')
	# ax2.set_xlabel('X label')
	# ax2.set_ylabel('Y label')
	# ax2.set_zlabel('Z label')
	#
	# km = KMeans(n_clusters=3,random_state=42)
	# result = km.fit_transform(X=train_data)
	# ax = fig.add_subplot(111, projection='3d')
	# # ax = Axes3D(fig)
	# for i in range(result1.shape[0]):
	# 	if train_label[i] == 0:
	# 		ax.scatter(result[i, 0], result[i, 1], result[i, 2], c='r')
	# 	elif train_label[i] == 1:
	# 		ax.scatter(result[i, 0], result[i, 1], result[i, 2], c='b')
	# 	else:
	# 		ax.scatter(result[i, 0], result[i, 1], result[i, 2], c='y')
	# ax.set_title('Kmeans')
	# ax.set_xlabel('X label')
	# ax.set_ylabel('Y label')
	# ax.set_zlabel('Z label')
	plt.show()

main()
