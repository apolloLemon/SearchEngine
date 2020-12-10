import nltk
import gensim
from operator import itemgetter
from nltk.tokenize import RegexpTokenizer
from bs4 import BeautifulSoup

import os
tokenizer = RegexpTokenizer(r'\w+')
def ProcessDocument(doc):
	sw = set(nltk.corpus.stopwords.words('english'))
	ps = nltk.stem.PorterStemmer()
    
	tk = tokenizer.tokenize(doc)

	return [ps.stem(t.lower()) for t in tk if t.lower() not in sw]

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
dictionary.save('test.dict')


vectors = [dictionary.doc2bow(pc) for pc in pC]
gensim.corpora.MmCorpus.serialize('test.mm', vectors)
mmcorpus = gensim.corpora.MmCorpus('test.mm')


model = gensim.models.TfidfModel(mmcorpus)

index = gensim.similarities.MatrixSimilarity(mmcorpus)


Q = "Document will identify a development in the Iran-Contra Affair"
pQ = ProcessDocument(Q)
vQ = dictionary.doc2bow(pQ)

wQ = model[vQ]
similarities = index[wQ]
ranking = sorted(enumerate(similarities), key=itemgetter(1), reverse=True)



for c, s in ranking[:10]:
    print("[ Score = %.3f" %s +" ] " + doc_list[c].find('docno').text)