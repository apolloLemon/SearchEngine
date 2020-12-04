import nltk
import gensim
from operator import itemgetter

def ProcessDocument(doc):
	sw = set(nltk.corpus.stopwords.words('english'))
	ps = nltk.stem.PorterStemmer()
	tk = nltk.tokenize.wordpunct_tokenize(doc)

	return [ps.stem(t.lower()) for t in tk if t.lower() not in sw]

C = [
	"This is a first example of a piece of text within a document from a corpus of documentation",
	"This second phrase is to add to the corpus and test the dictionary functionality",
	"The third string of characters is also to help test the dictionary",
	"Dictionaries are important, and so are vectors"
	]

pC = [ProcessDocument(c) for c in C]
dictionary = gensim.corpora.Dictionary(pC)
dictionary.save('test.dict')

vectors = [dictionary.doc2bow(pc) for pc in pC]
gensim.corpora.MmCorpus.serialize('test.mm', vectors)
mmcorpus = gensim.corpora.MmCorpus('test.mm')

model = gensim.models.TfidfModel(mmcorpus)
index = gensim.similarities.MatrixSimilarity(mmcorpus)


Q = "I'm looking for dictionaries"
pQ = ProcessDocument(Q)
vQ = dictionary.doc2bow(pQ)
wQ = model[vQ]
similarities = index[wQ]
ranking = sorted(enumerate(similarities), key=itemgetter(1), reverse=True)

print("Documents")
for c in C:
	print(c)

print("\nQuery : "+Q+"\n")

for c, s in ranking:
	print("[ Score = %.3f" %s +" ] " + C[c])