#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.word2vec import LineSentence
import cPickle
import os
import itertools

WIKI_DIR_PATH = "/Users/kumada/Data/enwiki"
WIKI_PATH = os.path.join(WIKI_DIR_PATH, "enwiki-{}-pages-articles.xml.bz2")
WIKI_CORPUS_PATH = os.path.join(WIKI_DIR_PATH, "{}-wiki-corpus.pkl")
WIKI_TEXT_PATH = os.path.join(WIKI_DIR_PATH, "{}.wiki")
WIKI_SENTENCE_PATH = os.path.join(WIKI_DIR_PATH, "{}_sentence.txt")


def write_wiki(corpus, output_path, titles=[]):
    with open(output_path, 'wb') as f:
        corpus.metadata = True
        for text, (page_id, title) in corpus.get_texts():
            if title not in titles:
                f.write(b' '.join(text) + b'\n')
                titles.append(title)
    return titles


def save_corpus(wiki_path, corpus_path):

    # make a corpus
    corpus = WikiCorpus(wiki_path)

    # save it
    print("output path: {}".format(corpus_path))
    cPickle.dump(corpus, open(corpus_path, "wb"))


if __name__ == "__main__":
    date_number = "20160920"
#    # make bz2 file path(input)
#    wiki_path = WIKI_PATH.format(date_number)
#    assert os.path.exists(wiki_path), ""

    # make a corpus path(output)
#    corpus_path = WIKI_CORPUS_PATH.format(date_number)
#    save_corpus(wiki_path, corpus_path)
#    corpus = cPickle.load(open(corpus_path))
#    print(corpus)

    # make a wiki path(output)
    wiki_path = WIKI_TEXT_PATH.format(date_number)
    print(wiki_path)
#    titles = write_wiki(corpus, wiki_path)
#    print(len(titles))
#    print(titles[:10])

    wiki_sentence = LineSentence(wiki_path)

    # words = set()
    for _, word in itertools.izip(range(3), wiki_sentence):
        print(len(word))
        # s = set(word)
        # words = words.union(s)

#    print(len(words))
