# laplacian = degree - adjacency

import numpy as np
import sys
import pdb


def largest_eig_vec(A):
    
    eig_vals, eig_vecs = np.linalg.eig(A)
    sort_perm = eig_vals.argsort()
    # pdb.set_trace()
    # sorts eig_vals and eig_vecs
    eig_vals.sort()     # <-- This sorts the list in place.
    # print(eig_vals)
    eig_vecs = np.real(eig_vecs[:, sort_perm])

    return eig_vecs[:,-1]

# def check_symmetric(a, rtol=1e-05, atol=1e-08):
#     return np.allclose(a, a.T, rtol=rtol, atol=atol)
# print(check_symmetric(A),sum(sum(A)))

if __name__ == "__main__":
    infile = sys.argv[1]
    outfile = sys.argv[2]

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

    # make transition_matrix
    
    trans_mat = np.zeros((num_nodes+1, num_nodes+1))
    for line in data:
        from_node, to_node = line.split()
        from_node = int(from_node)
        to_node = int(to_node)
        if to_node!=from_node:
            trans_mat[to_node,from_node] = 1.0

    # handle dead_ends to always teleport
    col_means = trans_mat.sum(axis=0)
    # print(col_means.shape)
    for i in range(col_means.shape[0]):
        if col_means[i] == 0:
            trans_mat[:,i] = 1.0
            # print(i)

    # make trans_matrix column stochastic
    trans_mat = trans_mat / trans_mat.sum(axis=0)

    # compute A matrix
    A = (0.8*trans_mat) + (0.2*(np.full((trans_mat.shape),1.0/(num_nodes+1))))       

    imp = largest_eig_vec(A)
    imp_nodes = imp.argsort()

    # Output 20 most important nodes
    
    with open (outfile,'w') as f:
        for i in range(1,21):
            f.write(str(imp_nodes[-i]) + "\n")
        
