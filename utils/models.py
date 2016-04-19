#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sqlalchemy.ext.declarative import declarative_base
import sqlalchemy as sa

Base = declarative_base()


class Article(Base):

    __tablename__ = "article"
    id = sa.Column("id", sa.Integer, primary_key=True)
    topics = sa.Column("topics", sa.String(100))
    date = sa.Column("date", sa.String(100))
    place = sa.Column("place", sa.String(100))
    title = sa.Column("title", sa.String(100))
    dateline = sa.Column("dateline", sa.String(100))
    body = sa.Column("body", sa.Text())


class StemmedArticle(Base):
    __tablename__ = "stemmed_article"
    id = sa.Column("id", sa.Integer, primary_key=True)
    article_id = sa.Column(sa.Integer, sa.ForeignKey('article.id'), nullable=False)
    body = sa.Column("body", sa.Text())
