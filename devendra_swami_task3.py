# laplacian = degree - adjacency

import numpy as np
from sklearn.cluster import KMeans
import sys


def k_small_eig_vecs(A,k,tol=1e-5):
    eig_vals, eig_vecs = np.linalg.eig(A)
    sort_perm = eig_vals.argsort()

    # sorts eig_vals and eig_vecs
    eig_vals.sort()     # <-- This sorts the list in place.
    eig_vecs = np.real(eig_vecs[:, sort_perm])

    lowest_pos = 0
    while(True):
        if eig_vals[lowest_pos]>=1e-5:
            break
        lowest_pos+=1
    return eig_vecs[:,lowest_pos:lowest_pos+k]

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)
# print(check_symmetric(A),sum(sum(A)))

if __name__ == "__main__":
    infile = sys.argv[1]
    outfile = sys.argv[2]
    k = int(sys.argv[3])

    # read Input
    with open (infile) as f:
        data = f.readlines()

    # find number of nodes
    num_nodes = 0
    for line in data:
        for node in line.split():
            node = int(node)
            if node>num_nodes:
                num_nodes = node

    # make laplacian_matrix
    A = np.zeros((num_nodes+1, num_nodes+1))
    for line in data:
        node1, node2 = line.split()
        node1 = int(node1)
        node2 = int(node2)
        if A[node1,node2]==0 and node1!=node2:
            A[node1,node2] = -1
            A[node2,node1] = -1
            A[node1, node1]+=1
            A[node2, node2]+=1

    num_components = 3 
    data = k_small_eig_vecs(A,num_components)
    kmeans = KMeans(n_clusters=k, random_state = 101)
    kmeans.fit(data)

    # Output predictions
    labels = kmeans.labels_
    with open (outfile,'w') as f:
        for i in range(len(labels)):
            f.write(str(i) + " " + str(labels[i]) + "\n")
        
