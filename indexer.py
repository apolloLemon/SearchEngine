import gensim
from bs4 import BeautifulSoup
import os
import docproc
import numpy as np

doc_list=[]
for filename in os.listdir(os.getcwd()+"/corpus"):
  with open(os.getcwd()+"/corpus/"+filename, 'r') as F:
    soup = BeautifulSoup(F,'html.parser')
    for doc in soup('doc'):
        doc_list.append(doc)
nplist = np.array(doc_list)
np.save('doclist.npy', nplist)

pC = []
for doc in doc_list :
    document_texts = ""
    for text in doc.find_all('text'):
        document_texts += text.text
    pC.append(docproc.ProcessDocument(document_texts))

    


dictionary = gensim.corpora.Dictionary(pC)
dictionary.save('corpus.dict')


vectors = [dictionary.doc2bow(pc) for pc in pC]
gensim.corpora.MmCorpus.serialize('corpus.mm', vectors)