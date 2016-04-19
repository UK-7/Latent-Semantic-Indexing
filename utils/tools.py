#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from sqlalchemy import create_engine
from bs4 import BeautifulSoup
from models import Article, StemmedArticle
from sqlalchemy.orm import sessionmaker
import textmining
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import EnglishStemmer
import string
import numpy as np
# from numpy import genfromtxt
import pandas as pd


def create_session():
    # create a connection to MySQL server.
    engine = create_engine('mysql://root:@localhost/article_search', echo=True)
    Session = sessionmaker(bind=engine)
    session = Session()
    return session


def article_parser(article):
    a = dict()
    for attr in ["title", "date", "topics", "dateline", "body"]:
        if article.find(attr):
            a[attr] = article.find(attr).text.strip()
        else:
            a[attr] = ""
    if article.find("places"):
        a["place"] = ",".join([p.text.strip()
                              for p in article.find("places").findAll("d")])
    a["body"] = a["body"].replace("Reuter", "").strip()
    return a


def remove_stopwords(s):
    # Remove stopwords and punctuation.
    stop = stopwords.words('english') + [str(p) for p in string.punctuation]
    token = [i for i in word_tokenize(s.lower()) if i not in stop]
    return token


def stemming(token):
    stemmer = EnglishStemmer()
    stemmed_str = " ".join(stemmer.stem(word) for word in token)
    return stemmed_str


def sgmparser(datafile):
    # Parse data from *.sgm
    session = create_session()
    with open(datafile, 'r') as f:
        data = f.read()
        soup = BeautifulSoup(data)
        articles = soup.findAll("reuters")
        for article in articles:
            parsed_article = article_parser(article)
            a = Article(**parsed_article)
            session.add(a)
        session.commit()


def store_stemmed_str(id, s):
    a = {
        "article_id": id,
        "body": s
    }
    new_stemmed_article = StemmedArticle(**a)
    return new_stemmed_article


def store_article():
    base_path = "dataset/"
    for i in range(0, 22):
        if i < 10:
            filename = "reut2-00" + str(i) + ".sgm"
        else:
            filename = "reut2-0" + str(i) + ".sgm"
        sgmparser(base_path + filename)


def term_document_matrix():
    session = create_session()
    tdm = textmining.TermDocumentMatrix()
    session = create_session()
    articles = session.query(Article.id, Article.body).all()
    for a in articles:
        #TODO Issue 5: cannot decode bug.
        try:
            token = remove_stopwords(a[1])
            id = int(a[0])
            s = stemming(token)
            # stemmed_a = store_stemmed_str(id, s)
            tdm.add_doc(s)
            # session.add(stemmed_a)
            # session.commit()
        except:
            continue
    tdm.write_csv('matrix1.csv', cutoff=1)


def create_weighted_matrix(td_matrix):
    ####################
    # Test
    ####################
    # m, n = 10, 10
    # a = np.array([e for e in range(1, 101)])
    # b = a.reshape((10, 10))
    ####################

    m, n = td_matrix.shape
    print m, n
    weighted_matrix = np.empty((m, n))
    gf = np.empty((1, m))

    # Compute gf, gf_i is the total number of times of term i
    for i in range(m):
        sum = 0
        for j in range(n):
            sum += td_matrix[i, j]
        gf[0, i] = sum

    print gf

    # gi is the global weight of term i.
    gi = np.empty((1, m))
    for i in range(m):
        sum = 0
        for j in range(n):
            p_ij = td_matrix[i, j]/gf[0, i]
            sum += (p_ij * np.log(p_ij))
        sum = sum/np.log(n)
        gi[0, i] = 1 + sum

    # Update t-d matrix by a_ij = gi * log(A(i, j) + 1)
    # https://en.wikipedia.org/wiki/Latent_semantic_indexing
    for i in range(m):
        for j in range(n):
            weighted_matrix[i, j] = gi[0, i] * np.log(td_matrix[i, j] + 1)
    return weighted_matrix

def create_gi_gfi(td_matrix):
    m, n = td_matrix.shape
    print m, n
    weighted_matrix = np.empty((m, n))
    gf = np.empty((1, m))

    # Compute gf, gf_i is the total number of times of term i
    for i in range(m):
        sum = 0
        for j in range(n):
            sum += td_matrix[i, j]
        gf[0, i] = sum


    # gi is the global weight of term i.
    gi = np.empty((1, m))
    for i in range(m):
        sum = 0
        for j in range(n):
            print "gf[0, i]=%d" %gf[0, i]
            p_ij = td_matrix[i, j]/gf[0, i]
            sum += (p_ij * np.log(p_ij))
        sum = sum/np.log(n)
        gi[0, i] = 1 + sum

    return gf, gi


if __name__ == "__main__":
    # term_document_matrix()
    data = pd.read_csv("term-document-matrix.csv")
    # Dimension (29116, 21577)
    # print create_gi_gfi(data.as_matrix())
    print create_gi_gfi(data.as_matrix())
