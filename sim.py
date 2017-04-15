import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import graph_gen

ALPHA = 0.7
BETA = 0.2

MAX_STEPS = 1000000
NUM_NODES = 7

RESOLUTION = MAX_STEPS / 10

# Budget probability function 
def h(x, y):
	return (x+y)/(x+y+BETA) * (x*x)/(x*x + y*y)


# Get initial awareness levels
def gen_initial_dist(N):
	A, B, SQ = np.zeros(N), np.zeros(N), np.zeros(N)
	for i in xrange(N):
		a, b = np.random.uniform(), np.random.uniform()
		if a > b:
			a, b = b, a

		A[i], B[i], SQ[i] = a, b-a, 1.0-b

	return A, B, SQ


# Assumes the budget allocation is uniform for now
def get_budget_dist(N):
	A = np.zeros((N, 3))
	for i in xrange(N):
		f1, f2 = 1.0/N, 1.0/N
		s1, s2 = h(f1, f2), h(f2, f1)
		A[i][0] = s1
		A[i][1] = s2
		A[i][2] = 1.0 - s1 - s2
	return A


# Samples index
def sample(A, i, N):
	r = np.random.uniform(high=A[i][-1])
	for j in range(N):
		if r < A[i][j]:
			return j

# Sample message
def sample_message(a, b, sq):
	s1, s2, s3 = a, a+b, a+b+sq
	r = np.random.uniform()
	if r < s1:
		return 'A'
	elif r < s2:
		return 'B'
	else:
		return 'SQ'


# Simulates single step
def simulate_step(N, E, A, B, SQ, BD, counts):
	i = np.random.randint(N)
	counts[i] += 1
	j = sample(E, i, N)

	r = np.random.uniform()
	# Peer group
	if r <= ALPHA:
		message = sample_message(A[j], B[j], SQ[j])
	else:
		# Firms
		message = sample_message(BD[i][0], BD[i][1], BD[i][2])

	if message == 'A':
		a1 = (A[i] * float(counts[i]) + 1) / (counts[i]+1)
		b1 = B[i] * float(counts[i]) / (counts[i]+1)
	elif message == 'B':
		a1 = A[i] * float(counts[i]) / (counts[i]+1)
		b1 = (B[i] * float(counts[i]) + 1) / (counts[i]+1)
	else:
		a1 = A[i] * float(counts[i]) / (counts[i]+1)
		b1 = B[i] * float(counts[i]) / (counts[i]+1)

	sq1 = 1.0 - a1 - b1
	A[i], B[i], SQ[i] = a1, b1, sq1
	return N, E, A, B, SQ, BD, counts


def simulate(verbose = True):
	N = NUM_NODES
	E = graph_gen.gen_random_ls_matrix(N)
	A, B, SQ = gen_initial_dist(N)
	BD = get_budget_dist(N)
	counts = np.zeros(N)

	for t in xrange(MAX_STEPS):
		if t % RESOLUTION == 0:
			print "t = " + str(t) + " ..."

		N, E, A, B, SQ, BD, counts = simulate_step(N, E, A, B, SQ, BD, counts)
		if verbose and t % RESOLUTION == 0:
			print "A:", A
			print "B:", B
			print "SQ:", SQ
			print ""

	print "Final Distribution:"
	print "A:", A
	print "B:", B
	print "SQ:", SQ
	print ""


if __name__ == '__main__':
	simulate(verbose = False)
