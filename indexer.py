import gensim
from bs4 import BeautifulSoup
import os
import docproc
import numpy as np

print("loading files")
doc_list=[]
for filename in os.listdir(os.getcwd()+"/corpus"):
  with open(os.getcwd()+"/corpus/"+filename, 'r') as F:
    soup = BeautifulSoup(F,'html.parser')
    for doc in soup('doc'):
        doc_list.append(doc)

print("prepoc files")
pC = []
for doc in doc_list :
    document_texts = ""
    for text in doc.find_all('text'):
        document_texts += text.text
    pC.append(docproc.ProcessDocument(document_texts))

    

print("make dict")
dictionary = gensim.corpora.Dictionary(pC)
dictionary.save('corpus.dict')

print("make vectors")
vectors = [dictionary.doc2bow(pc) for pc in pC]
gensim.corpora.MmCorpus.serialize('corpus.mm', vectors)

print("make model/index")
mmcorpus = gensim.corpora.MmCorpus('corpus.mm')
model = gensim.models.TfidfModel(mmcorpus)
index = gensim.similarities.MatrixSimilarity(mmcorpus)

print("save model/index")
model.save("corpus.model")
index.save("corpus.index")