import logging
from gensim import corpora, models, similarities, utils 

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]

# remove words that appear only once
all_tokens = sum(texts, [])
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
texts = [[word for word in text if word not in tokens_once] for text in texts]
print texts

dictionary = corpora.Dictionary(texts)
dictionary.save('/tmp/deerwester.dict')
print dictionary.token2id

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus) # store to disk, for later use
print corpus

lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=2)
doc1 = "Human computer interaction"
doc2 = "Relation of user perceived response time"
vec_bow1 = dictionary.doc2bow(doc1.lower().split())
vec_bow2 = dictionary.doc2bow(doc2.lower().split())
vec_lda = lda[[vec_bow1, vec_bow2]]

index = similarities.MatrixSimilarity(lda[corpus], similarity_type=utils.SimilariyType.COSINE)
sims = index[vec_lda] # perform a similarity query against the corpus
#print sorted(enumerate(sims), key=lambda item: -item[1])
for es in sims: 
    print sorted(enumerate(es), key=lambda item: -item[1])

index = similarities.MatrixSimilarity(lda[corpus], similarity_type=utils.SimilariyType.Negative_KL)
sims = index[vec_lda] # perform a similarity query against the corpus
#print sorted(enumerate(sims), key=lambda item: item[1])
for es in sims: 
    print sorted(enumerate(es), key=lambda item: item[1])

index = similarities.MatrixSimilarity(lda[corpus], similarity_type=utils.SimilariyType.KL)
sims = index[vec_lda] # perform a similarity query against the corpus
#print sorted(enumerate(sims), key=lambda item: item[1])
for es in sims: 
    print sorted(enumerate(es), key=lambda item: item[1])


