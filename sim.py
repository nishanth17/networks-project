import networkx as nx 
import numpy as np
import matplotlib.pyplot as plt

ALPHA = 0.4
BETA = 0.5

MAX_STEPS = 1000000
NUM_NODES = 5

# Budget probability function 
def h(x, y):
	return (x+y)/(x+y+BETA) * (x*x)/(x*x + y*y)


# Generates random left stochastic matrix such that the 
# principal diagonal entries are zero
def gen_random_ls_matrix(N):
	A = np.zeros((N, N))
	for i in xrange(N):
		r = np.random.uniform(size=N-2)
		r = np.sort(r)
		diffs = np.ediff1d(r)
		d0, dn = r[0], 1.0 - r[-1]
		if i == 0:
			A[1][0], A[N-1][0] = d0, dn
			A[2:-1, 0] = diffs
		elif i == N-1:
			A[0][N-1], A[N-2][N-1] = d0, dn
			A[1:N-2, N-1] = diffs
		else:
			A[0][i], A[N-1][i] = d0, dn
			A[1:i, i] = diffs[:i-1]
			A[i+1:N-1, i] = diffs[i-1:]

	for i in xrange(N):
		for j in xrange(1, N):
			A[i][j] += A[i][j-1]

	return A

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
def sample(A, i):
	r = np.random.uniform()
	for j in range(N):
		if A[i][j] < r:
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
	j = sample(E, i)

	r = np.uniform.random()
	# Peer group
	if r <= ALPHA:
		message = sample_message(A[j], B[j], SQ[j])
	else:
		# Firms
		message = sample_message(BD[i][0], BD[i][1], BD[1][2])

	if message == 'A':
		a1 = (A[i] * float(counts[i]-1) + 1) / counts[i]
		b1 = B[i] * float(counts[i]-1) / counts[i]
	elif message == 'B':
		a1 = A[i] * float(counts[i]-1) / counts[i]
		b1 = (B[i] * float(counts[i]-1) + 1) / counts[i]
	else:
		a1 = A[i] * float(counts[i]-1) / counts[i]
		b1 = B[i] * float(counts[i]-1) / counts[i]

	sq1 = 1.0 - a1 - b1
	A[i], B[i], SQ[i] = a1, b1, sq1
	return N, E, A, B, SQ, BD, counts


def simulate():
	N, E = NUM_NODES, gen_random_ls_matrix(N)
	A, B, SQ = gen_initial_dist(N)
	BD = get_budget_dist(N)
	counts = np.zeros(N)

	for t in xrange(MAX_STEPS):
		N, E, A, B, SQ, BD, counts = simulate_step(N, E, A, B, SQ, BD, counts)

	# TODO: Analyze distributions
	return 0


# TODO: Something 
if __name__ == '__main__':
	pass