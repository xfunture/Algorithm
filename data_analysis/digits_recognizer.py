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
import tensorflow as tf
from keras.utils import to_categorical
from time import sleep
import os
from keras.datasets import mnist
from keras.layers import Dense,Conv2D,Dropout,BatchNormalization,MaxPool2D,Flatten,Input
import seaborn as sea
import warnings


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
warnings.filterwarnings('ignore')


def load_digits_data():
	# data = load_digits(n_class=10)
	# train_data = data['data']
	# train_label = data['target']
	# print(type(train_data))
	# print(train_data.shape)
	#
	X,y = mnist.load_data()
	X_train_mnist,y_train_mnist = X[0],X[1]
	X_test_mnist,y_test_mnist = y[0],y[1]
	print(X_train_mnist.shape,X_test_mnist.shape)
	df = pd.read_csv('digit-recognizer/train.csv',sep=',',index_col=None)
	df = df.sample(frac = 1,random_state=42)
	y = df['label'].to_numpy()
	columns = df.columns.tolist()
	X = df.loc[:,columns[1:]]
	X = X.to_numpy()
	X = X.reshape(len(X),28,28)
	print(X.shape,y.shape)
	X = np.concatenate([X,X_train_mnist,X_test_mnist])
	y = np.concatenate([y,y_train_mnist,y_test_mnist])
	X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.01,random_state=42)
	print('X_train shape:{0}'.format(X_train.shape))
	print('X_val shape:{0}'.format(X_val.shape))
	print('y_train shape:{0}'.format(y_train.shape))
	print('y_val shape:{0}'.format(y_val.shape))

	return 	X_train, X_val, y_train, y_val


def load_test():
	df = pd.read_csv('digit-recognizer/test.csv',sep=',',index_col = None)
	df = df.to_numpy()
	df = df.reshape(df.shape[0],28,28)
	print(df.shape)
	return df


def load_mnist_digits():
	load_digits()
	
def load_data_v3():
	df = pd.read_csv('digit-recognizer/train.csv', sep=',', index_col=None)
	df = df.sample(frac=1, random_state=42)
	y = df['label'].to_numpy()
	columns = df.columns.tolist()
	X = df.loc[:, columns[1:]]
	X = X.to_numpy()
	X = X.reshape(len(X), 28, 28)
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01, random_state=42)
	print('X_train shape:{0}'.format(X_train.shape))
	print('X_val shape:{0}'.format(X_val.shape))
	print('y_train shape:{0}'.format(y_train.shape))
	print('y_val shape:{0}'.format(y_val.shape))
	return X_train, X_val, y_train, y_val

def build_network(X_train,y_train,X_val,y_val,X_test):
	epochs = 2
	batch_size = 64
	X = tf.placeholder(tf.float32,shape=(None,28,28),name='X')
	y = tf.placeholder(tf.int32,shape=(None),name='y')
	layer1 = tf.layers.conv1d(inputs=X,filters=128,kernel_size=(8,),activation='relu',strides=1,padding='valid')
	layer2 = tf.layers.conv1d(inputs=layer1,filters=128,kernel_size=(8,),activation='relu',strides=1,padding='valid')
	layer3 = tf.layers.max_pooling1d(inputs=layer2,pool_size=4,strides = 1)
	layer4 = tf.layers.conv1d(inputs=layer3,filters=64,kernel_size=(8,),activation='relu',strides=1,padding='valid')
	layer6 = tf.layers.max_pooling1d(inputs=layer4,pool_size=4,strides = 1)
	# layer4 = tf.layers.conv1d(inputs=layer3,filters=128,kernel_size=(8,),strides=1,padding='valid')
	# layer5 = tf.layers.conv1d(inputs=layer4,filters=64,kernel_size=(4,),strides=1,padding='valid')
	flatten = tf.layers.flatten(layer6)
	dropout = tf.layers.dropout(inputs=flatten)
	logits = tf.layers.dense(inputs = dropout,units=10)
	predict = tf.nn.softmax(logits=logits,axis=1)
	predict = tf.arg_max(logits,1)
	loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=y))
	train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	# output1,output2,output3 = sess.run([loss,logits,predict],feed_dict={X:X_train[:batch_size,:,:],y:y_train[:batch_size]})
	# print(output1,output2.shape)
	# print(output3.shape,output3)
	for j in range(epochs):
		for i in range(0,X_train.shape[0],batch_size):
			if i+batch_size>X_train.shape[0]:
				break
			_,loss_value,y_predict = sess.run([train_op,loss,predict],feed_dict={X:X_train[i:i+batch_size],y:y_train[i:i+batch_size]})
			accracy = metrics.accuracy_score(y_train[i:i+batch_size],y_predict)
			# print('train loss:{0} accuracy:{1}'.format(loss_value,accracy))			# print('train loss:{0} accuracy:{1}'.format(loss_value,accracy))
		print('train loss:{0} accuracy:{1}'.format(loss_value,accracy))
		_, loss_value, y_predict = sess.run([train_op, loss, predict],feed_dict={X: X_val, y: y_val})
		accracy = metrics.accuracy_score(y_val, y_predict)
		print('epoch:{2} val loss:{0} accuracy:{1}'.format(loss_value, accracy,j))
		sleep(2)


	y_predict = sess.run(predict, feed_dict={X:X_test})
	df = pd.DataFrame({'ImageId':range(1,X_test.shape[0]+1),'Label':y_predict})
	df.to_csv('digit-recognizer/sample_submission.csv',sep=',',index=None)
	
	
def build_network_v2(X_train,y_train,X_val,y_val,X_test):
	epochs = 70
	batch_size = 64
	X = tf.placeholder(tf.float32,shape=(None,28,28),name='X')
	y = tf.placeholder(tf.int32,shape=(None),name='y')
	
	layer1 = tf.layers.conv1d(inputs=X,filters=32,kernel_size=(3,),activation='relu',strides=1,padding='valid')
	ba1 = tf.layers.batch_normalization(inputs=layer1)
	
	layer2 = tf.layers.conv1d(inputs=ba1,filters=32,kernel_size=(3,),activation='relu',strides=1,padding='valid')
	ba2 = tf.layers.batch_normalization(inputs=layer2)
	
	layer3 = tf.layers.conv1d(inputs=ba2, filters=32, kernel_size=(5,), activation='relu', strides=2,padding='same')
	ba3 = tf.layers.batch_normalization(inputs=layer3)
	
	dropout1 = tf.layers.dropout(inputs=ba3)
	
	layer4 = tf.layers.conv1d(inputs=dropout1, filters=64, kernel_size=(3,), activation='relu', strides=1, padding='valid')
	ba4 = tf.layers.batch_normalization(inputs=layer4)
	
	layer5 = tf.layers.conv1d(inputs=ba4, filters=64, kernel_size=(3,), activation='relu', strides=1, padding='valid')
	ba5 = tf.layers.batch_normalization(inputs=layer5)
	
	layer6 = tf.layers.conv1d(inputs=ba5, filters=64, kernel_size=(5,), activation='relu', strides=2, padding='same')
	ba6 = tf.layers.batch_normalization(inputs=layer6)
	
	dropout2 = tf.layers.dropout(inputs=ba6)
	
	layer7 = tf.layers.conv1d(inputs=dropout2, filters=128, kernel_size=(4,), activation='relu', strides=1, padding='valid')
	ba7 = tf.layers.batch_normalization(inputs=layer7)
	
	flatten = tf.layers.flatten(ba7)
	dropout = tf.layers.dropout(inputs=flatten)
	logits = tf.layers.dense(inputs = dropout,units=10)
	predict = tf.nn.softmax(logits=logits,axis=1)
	predict = tf.arg_max(logits,1)
	loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=y))
	train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	for j in range(epochs):
		for i in range(0,X_train.shape[0],batch_size):
			if i+batch_size>X_train.shape[0]:
				break
			_,loss_value,y_predict = sess.run([train_op,loss,predict],feed_dict={X:X_train[i:i+batch_size],y:y_train[i:i+batch_size]})
			accracy = metrics.accuracy_score(y_train[i:i+batch_size],y_predict)
			# print('train loss:{0} accuracy:{1}'.format(loss_value,accracy))			# print('train loss:{0} accuracy:{1}'.format(loss_value,accracy))
		
		
		print('train loss:{0} accuracy:{1}'.format(loss_value,accracy))
		_, loss_value, y_predict = sess.run([train_op, loss, predict],feed_dict={X: X_val, y: y_val})
		accracy = metrics.accuracy_score(y_val, y_predict)
		print('epoch:{2} val loss:{0} accuracy:{1}'.format(loss_value, accracy,j))
		sleep(2)


	y_predict = sess.run(predict, feed_dict={X:X_test})
	df = pd.DataFrame({'ImageId':range(1,X_test.shape[0]+1),'Label':y_predict})
	df.to_csv('digit-recognizer/sample_submission.csv',sep=',',index=None)

def build_network_v1(X_train,y_train,X_val,y_val,X_test):
	# X_train = X_train.reshape(len(X_train),28,28,1)
	# y_train = tf.keras.utils.to_categorical(y_train,num_classes=10)
	# y_val = tf.keras.utils.to_categorical(y_train,num_classes=10)
	
	
	model = tf.keras.models.Sequential()
	model.add(tf.keras.Input(shape=(28,28)))
	model.add(tf.keras.layers.Conv1D(filters=32,kernel_size=(3),activation='relu',padding='valid',strides=1))
	model.add(tf.keras.layers.BatchNormalization())

	model.add(tf.keras.layers.Conv1D(filters=32,kernel_size=(3),activation='relu',padding='valid',strides=1))
	model.add(tf.keras.layers.BatchNormalization())

	model.add(tf.keras.layers.Conv1D(filters=32,kernel_size=(5),activation='relu',padding='valid',strides=2))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.Dropout(rate=0.5))
	
	model.add(tf.keras.layers.Conv1D(filters=64,kernel_size=(3),activation='relu',padding='valid',strides=1))
	model.add(tf.keras.layers.BatchNormalization())

	model.add(tf.keras.layers.Conv1D(filters=64,kernel_size=(3),activation='relu',padding='valid',strides=1))
	model.add(tf.keras.layers.BatchNormalization())

	model.add(tf.keras.layers.Conv1D(filters=64,kernel_size=(5),activation='relu',padding='same',strides=2))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.Dropout(rate=0.5))

	# model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=(4), activation='relu', padding='valid', strides=1))
	# model.add(tf.keras.layers.BatchNormalization())
	
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
	model.compile(optimizer='adam',loss="sparse_categorical_crossentropy",metrics=['accuracy'])
	tf.keras.utils.plot_model(model,to_file='models.png',show_shapes=True)
	model.fit(x=X_train,y=y_train,epochs=40)
	test_loss,test_acc = model.evaluate(x=X_val,y=y_val)
	y_predict = model.predict(X_test)
	y_predict = np.argmax(y_predict,axis=1)
	print('Test Accuracy:{0}'.format(test_acc))
	print(y_predict.shape)
	print(y_predict[:10])
	df = pd.DataFrame({'ImageId':np.arange(1,len(X_test)+1),'Label':y_predict})
	df.to_csv('digit-recognizer/sample_submission.csv',sep=',',index=None)
	
def build_network_v3(X_train,y_train,X_val,y_val,X_test):
	
	# model = tf.models.Sequential()
	model = tf.keras.models.Sequential()
	model.add(tf.keras.Input(shape=(28,28,1)))
	model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=(5,5),padding='Same',activation='relu',input_shape=(28,28,1)))
	# model.add(BatchNormalization())
	#
	# model.add(Conv2D(filters=64,kernel_size=(5,5),padding='Same',activation='relu'))
	# model.add(BatchNormalization())
	#
	#
	# model.add(MaxPool2D(pool_size=(2,2)))
	# model.add(Dropout(rate=0.25))
	#
	# model.add(Conv2D(filters=64,kernel_size=(3,3),padding='Same',activation='relu'))
	# model.add(BatchNormalization())
	#
	# model.add(Conv2D(filter=64,kernel_size=(3,3),padding='Same',activation='relu'))
	# model.add(BatchNormalization())
	# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
	# model.add(Dropout(0.25))
	#
	# model.add(Conv2D(filters=64,kernel_size=(3,3),padding='Same',activation='relu'))
	# model.add(BatchNormalization())
	# model.add(Dropout(0.25))
	#
	# model.add(Flatten())
	# model.add(Dense(256,activation='relu'))
	# model.add(BatchNormalization())
	# model.add(Dropout(0.25))
	# model.add(Dense(10,activation='softmax'))
	
	
	

def main():
# 	X_train, X_val, y_train, y_val = load_digits_data()
# 	X_test = load_test()
# 	build_network_v2(X_train,y_train,X_val,y_val,X_test)

	X_train, X_val, y_train, y_val = load_data_v3()
	X_test = load_test()
	build_network_v1(X_train,y_train,X_val,y_val,X_test)
	# build_network_v3(X_train,y_train,X_val,y_val,X_test)
	# fig = plt.figure(figsize=(20,4))
	# sea.countplot(y_val)
	# plt.show()

if __name__ == "__main__":
	main()
