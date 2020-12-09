import nltk
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

def ProcessDocument(doc):
	sw = set(nltk.corpus.stopwords.words('english'))
	ps = nltk.stem.PorterStemmer()
    
	tk = tokenizer.tokenize(doc)

	return [ps.stem(t.lower()) for t in tk if t.lower() not in sw]