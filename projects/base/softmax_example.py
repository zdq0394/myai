import numpy as np
import math


# in_matrix = [1,2,3]
def sofx_max(in_matrix):
    m,n = np.shape(in_matrix)
    out_matrix = np.mat(np.zeros((m, n)))
    smx = 0.0
    for j in range(n):
        out_matrix[0, j] = math.exp(in_matrix[0, j])
        smx += out_matrix[0, j]

    for j in range(n):
        out_matrix[0, j] = out_matrix[0, j] / smx

    return out_matrix


test_mat = np.mat([1, 2, 3])
print(sofx_max(test_mat))
