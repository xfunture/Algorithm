import numpy as np
import collections

from collections import OrderedDict,Sequence,MutableSequence

import time



def fibi(N):
	if (N==1 or N==2):
		return 1
	record = {1:1,2:1}
	return helper(N,record)

def helper(N,record):
	if N in record:
		return record[N]
	else:
		value = helper(N-1,record) + helper(N-2,record)
		record[N] = value
		return value
	
def fibi_old(N):
	if (N==1 or N==2):
		return 1
	return fibi_old(N-1) + fibi_old(N-2)


def main():
	value = fibi(20)
	# value = fibi_old(44)
	print(value)

if __name__ == "__main__":
	main()
	