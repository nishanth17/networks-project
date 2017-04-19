import numpy as np

def compute_row_prefix_sums(A):
	N = len(A)
	for i in xrange(N):
		for j in xrange(1, N):
			A[i][j] += A[i][j-1]
	return A


def round(A, small):
	for i in xrange(len(A)):
		if A[i] < small:
			A[i] = small
	return A