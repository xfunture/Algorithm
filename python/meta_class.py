#coding:utf-8

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


class MyList(list,metaclass=ListMetaclass):
	pass
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
	L = MyList()
	L.append(1)
	L.add(2)
	print(L)


main()
