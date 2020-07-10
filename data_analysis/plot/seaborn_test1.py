import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def analysis():
	df = pd.read_csv('../titanic/train.csv',sep=',')
	print(df.columns)
	# print(df.isnull().sum())
	# sns.distplot(a=df['Fare'])
	# sns.factorplot(x='Pclass',y='Survived',data=df,kind='bar')
	# sns.barplot(x='Pclass',y='Survived',data=df)
	plt.show()








analysis()
