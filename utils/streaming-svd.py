from __future__ import division
from scipy.sparse.linalg import svds
from numpy.linalg import svd

import pandas as pd
import numpy as np
import math

np.set_printoptions(threshold=np.nan)

'''
Parse input csv file to create document-term matrix as an NP array
Input: path to term-document matrix
Output: matrix A, term-document matrix
'''
def readData(path):
      A = []
      with open(path) as f:
            for index, line in enumerate(f):
                  if index >= 5:
                        data.append(line.strip().split(","))
            A = np.array(A, dtype="i")
      return A

'''
Parse the rows of B to find the first zero row
Input: Matrix B as np array
Output: index of first 0 row. -1 if no such row present
'''
def getZeroRows(B):
      m, n = B.shape
      zeroRows=[]
      for i in range(m):
            if np.sum(B[i,:]) == 0:
                  zeroRows.append(i)
      return zeroRows

'''
Frequent-Directions: creates a matrix-sketch of number of rows
specified by l and smae number of cilumns as A
Input: Term-Document matrix A and desired number of columns in sketch l
Output: Sketch of matrix A
'''
def frequentDirections(path, l):
      B = np.zeros(l)
      with open(path) as f:
            zeroRows = range(l)
            print type(zeroRows)
            for index, line in enumerate(f):
                  line = line.strip().split(",")
                  line = np.array(line)
                  if index == 0:
                        n = len(line)
                        B = np.zeros((l,n))
                        continue
                  line = line[1:]
                  if  len(zeroRows) == 0:
                        u, sigma, v =  svd(B, full_matrices=False)
                        delta = sigma[l/2]**2
                        sigma_new = [0.0 if d < 0.0 else math.sqrt(d) \
                                    for d in (sigma**2 - delta)]
                        B = np.dot(np.diagflat(sigma_new), v)
                        zeroRows = getZeroRows(B)
                  else:
                        B[zeroRows[0],:] = np.array(line)
                        zeroRows.remove(zeroRows[0])
            return B

if __name__ == "__main__":
      B = frequentDirections("term-document-matrix.csv", 1000)
      print np.sum(B)
   
