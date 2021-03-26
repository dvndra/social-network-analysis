# laplacian = degree - adjacency

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
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
    trainfile = sys.argv[2]
    testfile = sys.argv[3]
    outfile = sys.argv[4]

    # read Input edges file
    with open (infile) as f:
        data = f.readlines()

    # find number of nodes
    num_nodes = 0
    for line in data:
        for node in line.split():
            node = int(node)
            if node>num_nodes:
                num_nodes = node

    # make laplacian_matrix based on edges
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

    num_components = min(67,A.shape[0]/2)
    data = k_small_eig_vecs(A,num_components)
        
    # Read Train and test indices and train labels
    with open (trainfile) as f:
        temp = f.readlines()
    train_indices = []
    train_labels = []
    for line in temp:
        i, label = line.split()
        train_indices.append(int(i))
        train_labels.append(int(label)) 

    with open (testfile) as f:
        temp = f.readlines()
    test_indices = []
    for line in temp:
        i, _ = line.split()
        test_indices.append(int(i))              

    # Fit Model
    num_neighbors = 6 
    knn = KNeighborsClassifier(n_neighbors = num_neighbors)
    knn.fit(data[train_indices],train_labels)

    # Output predictions
    predictions = knn.predict(data[test_indices])
    with open (outfile,'w') as f:
        for i in range(len(predictions)):
            f.write(str(test_indices[i]) + " " + str(predictions[i]) + "\n")