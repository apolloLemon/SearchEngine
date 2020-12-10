import gensim
from operator import itemgetter
import docproc
from bs4 import BeautifulSoup
import os

dictionary = gensim.corpora.Dictionary()
dictionary.load('corpus.dict')

mmcorpus = gensim.corpora.MmCorpus('corpus.mm')

model = gensim.models.TfidfModel(mmcorpus)
index = gensim.similarities.MatrixSimilarity(mmcorpus)

doc_list=np.load('doclist.npy')

Q = input("Query: ")
pQ = docproc.ProcessDocument(Q)
vQ = dictionary.doc2bow(pQ)

wQ = model[vQ]
similarities = index[wQ]
ranking = sorted(enumerate(similarities), key=itemgetter(1), reverse=True)



for c, s in ranking[:10]:
    print("[ Score = %.3f" %s +" ] " + doc_list[c].find('docno').text)