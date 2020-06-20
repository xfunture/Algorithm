#coding:utf-8
class magic(object):
	def __init__(self,info):
		self.info = info
	
	def __getattr__(self, item):
		return '你想让我干啥子嘛'

	
	# def __getattribute__(self, item):
	# 	return '你想让我干啥子嘛'
	
def main():
	a = magic({'user':'浪子','age':18})
	print(a.age)
	print(a.ageaa)
	print(a.user)

main()