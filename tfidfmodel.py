#!/usr/bin/env python2.5
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


import logging
import itertools

import math

import interfaces
import matutils



class TfidfModel(interfaces.TransformationABC):
    """
    Objects of this class realize the transformation between word-document co-occurence
    matrix (integers) into a locally/globally weighted matrix (positive floats).
    
    This is done by combining the term frequency counts (the TF part) with inverse
    document frequency counts (the IDF part), optionally normalizing the resulting
    documents to unit length.
    
    The main methods are:
    1) constructor, which calculates IDF weights for all terms in the training corpus.
    
    2) the [] method, which transforms a simple count representation into the TfIdf 
    space.
    
    Model persistency is achieved via its load/save methods.
    """
    def __init__(self, corpus, id2word = None, normalize = True):
        """
        Find latent space based on the corpus provided.

        numTopics is the number of requested factors (latent dimensions).
        normalize dictates whether the resulting vectors will be set to unit length.
        
        After the model has been initialized, you can estimate topics for an
        arbitrary, unseen document, using the topics = self[bow] dictionary notation.
        """
        self.id2word = id2word
        self.normalize = normalize
        if corpus is not None:
            self.initialize(corpus)

    
    def __str__(self):
        return "TfidfModel(numTerms=%s)" % (self.numTerms)


    def initialize(self, corpus):
        """
        Compute inverse document weights, which will be used to modify term 
        frequencies for documents.
        """
        if self.id2word is None:
            logging.info("no word id mapping provided; initializing from corpus, assuming identity")
            maxId = 0
            for document in corpus:
                maxId = max(maxId, max([-1] + [fieldId for fieldId, _ in document]))
            self.numTerms = 1 + maxId
            self.id2word = dict(zip(xrange(self.numTerms), xrange(self.numTerms)))
        else:
            self.numTerms = 1 + max(self.id2word.iterkeys())
        
        logging.info("calculating IDF weights over %i documents" % len(corpus))
        idfs = {}
        numNnz = 0
        for docNo, bow in enumerate(corpus):
            if docNo % 5000 == 0:
                logging.info("PROGRESS: processing document %i/%i" % 
                             (docNo, len(corpus)))
            numNnz += len(bow)
            for termId, termCount in bow:
                idfs[termId] = idfs.get(termId, 0) + 1
        idfs = dict((termId, math.log(1.0 * (docNo + 1) / docFreq, 2)) # the IDF weight formula 
                    for termId, docFreq in idfs.iteritems())
        
        self.idfs = idfs
        
        # keep some stats about the training corpus
        self.numDocs = len(corpus)
        self.numNnz = numNnz


    def __getitem__(self, bow):
        """
        Return tf-idf representation of the input vector.
        
        This is done by folding the input document into the latent topic space.
        """
        # unknown (new) terms will be given zero weight (NOT infinity/huge weight,
        # as would the strict application of the IDF formula suggest
        vector = [(termId, tf * self.idfs.get(termId, 0.0)) 
                  for termId, tf in bow if self.idfs.get(termId, 0.0) != 0.0]
        if self.normalize:
            vector = matutils.unitVec(vector)
        return vector
#endclass TfidfModel
