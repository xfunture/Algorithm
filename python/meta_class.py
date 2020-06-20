# coding:utf-8

# class ListMetaclass(type):
# 	def __new__(cls, name, bases, attrs):
# 		attrs['add'] = lambda self, value: self.append(value)
# 		return type.__new__(cls, name, bases, attrs)
#
#
# class MyList(list):
# 	__metaclass__ = ListMetaclass  # 指示使用ListMetaclass 来定制类


class ListMetaclass(type):
	def __new__(cls, name, bases, attrs):
		attrs['add'] = lambda self, value: self.append(value)
		return type.__new__(cls, name, bases, attrs)


class MyList(list, metaclass=ListMetaclass):
	pass


class Demo(object):
	def __new__(cls, *args, **kwargs):
		print(cls, type(cls))
		return super(Demo, cls).__new__(cls, *args, **kwargs)


class Demo1(object):
	def __new__(cls, class_name, class_bases):
		print(cls, type(cls))
		return super(Demo1, cls).__new__(cls, class_name, class_bases)


class UpperAttrMetaClass(type):
	def __new__(mcs, class_name, class_parents, class_attr):
		print('mcs',mcs,type(mcs))
		print('class_name',class_name)
		print('class_parents',class_parents)
		print('class_attr',class_attr)
		attrs = ((name, value) for name, value in class_attr.items() if not name.startswith('__'))
		uppercase_attrs = dict((name.upper(), value) for name, value in attrs)
		return super(UpperAttrMetaClass, mcs).__new__(mcs, class_name, class_parents, uppercase_attrs)


class Trick(object,metaclass=UpperAttrMetaClass):
	bar = 12
	money = 'unlimited'


class Magic:
	def __init__(self,info):
		self.info = info
		
	def __getattr__(self, item):
		return self.info[item]


#
# def fn(self, name='world'):
# 	print('Hello, %s' % name)


# def test():
# 	# create class by type
# 	# first arg:class name
# 	# second arg:father class
# 	# third arg: class function name combine with the function fn
# 	Hello = type('Hello', (object,), dict(hello=fn))
# 	h = Hello()
# 	print(h.hello())
# 	print(type(Hello))
# 	print(type(h))


def main():
	# L = MyList()
	# L.append(1)
	# L.add(2)
	# print(L)
	# print(L)
	# demo = Demo1()
	# print(demo)
	# print(Trick.BAR)
	# print(Trick.MONEY)
	magic = Magic({'age':28,'name':'rick'})
	print(magic)
	print(magic.age)

main()
