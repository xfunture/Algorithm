from enum import Enum,unique

@unique
class Weekday(Enum):
	sum = 1
	Mon = 2
	Tue = 3
	Wed = 4
	Thu = 5
	Fri = 6
	Sat = 7

class Student(object):
	
	@property
	def name(self):
		return self._name
	
	@name.setter
	def name(self, value):
		self._name = value
	
	@property
	def age(self):
		return self._age
	
	@age.setter
	def age(self, value):
		self._age = value
	
	@property
	def score(self):
		return self._score
	
	@score.setter
	def score(self, value):
		self._score = value
		
	def __str__(self):
		return 'Student object name: %s' % self._name
	
	def __repr__(self):
		return 'Student object name； %s' % self._name


class Screen(object):
	@property
	def width(self):
		return self._width
	
	@width.setter
	def width(self, value):
		self._width = value
	
	@property
	def height(self):
		return self._height
	
	@height.setter
	def height(self, value):
		self._height = value
	
	@property
	def resolution(self):
		return self._width * self._height


class Fib(object):
	def __init__(self):
		self.a,self.b = 0,1
	
	def __iter__(self):
		return self
	
	def __next__(self):
		self.a,self.b = self.b,self.a + self.b
		if self.a > 100:
			raise StopIteration()
		return self.a
	

def main():
	s = Student()
	s.name = 'rick'
	print(s)
	s = Screen()
	s.width = 1024
	s.height = 768
	print('width =',s.width)
	print('height =',s.height)
	print('resolution =', s.resolution)
	if s.resolution == 786432:
		print('测试通过!')
	else:
		print('测试失败!')


main()
