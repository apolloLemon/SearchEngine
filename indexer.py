import gensim
from bs4 import BeautifulSoup
import os
import docproc

doc_list=[]
for filename in os.listdir(os.getcwd()+"/corpus"):
  with open(os.getcwd()+"/corpus/"+filename, 'r') as F:
    soup = BeautifulSoup(F,'html.parser')
    for doc in soup('doc'):
        doc_list.append(doc)
    

pC = []

for doc in doc_list :
    document_texts = ""
    for text in doc.find_all('text'):
        document_texts += text.text
    pC.append(ProcessDocument(document_texts))

    


dictionary = gensim.corpora.Dictionary(pC)
dictionary.save('corpus.dict')


vectors = [dictionary.doc2bow(pc) for pc in pC]
gensim.corpora.MmCorpus.serialize('corpus.mm', vectors)