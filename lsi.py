#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils.tools import create_weighted_matrix
from numpy.linalg import svd, inv
import numpy as np
from scipy.spatial.distance import cosine
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import EnglishStemmer
import string
from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfTransformer


class LSI(object):

    def __init__(self, n_dimension, word_dictionary):
        self.n_dimension = n_dimension
        self.word_dictionary = word_dictionary

    def _local_global_weight_matrix(self, td_matrix):
        # tools.create_weighted_matrix
        return create_weighted_matrix(td_matrix)

    def _truncatedSVD(self, td_matrix):
        # use scikit.truncatedSVD in case of not working here.
        # numpy.linalg.svd
        # weighted_matrix = self._local_global_weight_matrix(td_matrix)
        # print weighted_matrix
        # U, s, V = svd(weighted_matrix)
        # svd = TruncatedSVD(n_components=5, random_state=42)
        # tf_idf_matrix = TfidfTransformer()
        # print tf_idf_matrix.transform(td_matrix)

        td_matrix = td_matrix.astype(np.float)
        U, s, V = svds(td_matrix, k=15)

        self.s_truncated = s
        self.U_truncated = U
        self.V_truncated = V

        # self.s_truncated = s[:self.n_dimension]
        # self.U_truncated = U[:, :self.n_dimension]
        # self.V_truncated = V[:self.n_dimension, :]

    def _define_query(self, query):
        # return a term-frequency vector accoring to the order of words in
        # term-document matrix.
        vector = []
        stop = stopwords.words('english') + [str(p) for p in string.punctuation]
        token = [i for i in word_tokenize(query.lower()) if i not in stop]
        stemmer = EnglishStemmer()
        stemmed_str = [stemmer.stem(word) for word in token]
        for word in self.word_dictionary:
            if word in token:
                vector.append(1)
            else:
                vector.append(0)
        vector = np.array(vector)
        q = np.dot(np.dot(vector.reshape((1, vector.shape[0])), self.U_truncated), inv(np.diag(self.s_truncated)))
        return q

    def match(self, td_matrix, query):
        m, n  = td_matrix.shape
        self._truncatedSVD(td_matrix)
        q = self._define_query(query)
        print q
        max_similarity = []
        matching_idx = []
        for doc_idx in range(n):
            doc_vector = self.V_truncated[:, doc_idx]
            print doc_vector
            sim = cosine(q, doc_vector)
            print "======="
            print "Sim and Idx"
            print sim
            print doc_idx
            print "End"
            print "======="
            if len(max_similarity) == 5:
                if sim > min(max_similarity):
                    min_idx = np.argmin(max_similarity)
                    max_similarity[min_idx] = sim
                    matching_idx[min_idx] = doc_idx
            else:
                max_similarity.append(sim)
                matching_idx.append(doc_idx)
        return max_similarity, matching_idx


if __name__ == "__main__":
    import pandas as pd
    data = pd.read_csv("utils/term-document-matrix2.csv")
    data.index = data["term"]
    lsi = LSI(n_dimension = 100, word_dictionary = data.index)
    del data["term"]
    print data.as_matrix()
    print lsi.match(data.as_matrix(), "mcDonnell Douglas Corp CF6-80C2 engines")
