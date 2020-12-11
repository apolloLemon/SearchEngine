import gensim
from operator import itemgetter
import docproc
from bs4 import BeautifulSoup
import os

def searchDoc(query):
 
  print("loading files")
  doc_list=[]
  for filename in os.listdir(os.getcwd()+"/corpus"):
    with open(os.getcwd()+"/corpus/"+filename, 'r') as F:
      soup = BeautifulSoup(F,'html.parser')
      for doc in soup('doc'):
          doc_list.append(doc)

  print("load dict")
  dictionary = gensim.corpora.Dictionary().load('corpus.dict')
  #dictionary.load('corpus.dict')

  mmcorpus = gensim.corpora.MmCorpus('corpus.mm')

  print("load model/index")
  model = gensim.models.TfidfModel.load("corpus.model")
  index = gensim.similarities.MatrixSimilarity.load("corpus.index")

  #doc_list=np.load('doclist.npy')

  #Q = input("Query: ")
  Q=query
  pQ = docproc.ProcessDocument(Q)
  vQ = dictionary.doc2bow(pQ)

  wQ = model[vQ]
  similarities = index[wQ]
  ranking = sorted(enumerate(similarities), key=itemgetter(1), reverse=True)
  result = []
  texts = []
  for c, s in ranking[:10]:
      result.append(doc_list[c])
      texts.append(doc_list[c].find('text').text)
      print("[ Score = %.3f" %s +" ] " + doc_list[c].find('docno').text)
  return result, texts