import nltk
import torch.nn
from nltk.corpus import stopwords
from nltk import word_tokenize
import re
from imdb_vocab import Vocab
from gensim.models import Word2Vec

sen1 = []
sen2 = []
clean = ['.', '?']
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sentences = 'Shall I compare thee to a summer day? Thou art more lovely and more temperate. Rough winds do shake the darling buds of May. And summer lease hath all too short a date. Sometime too hot the eye of heaven shines. And often is his gold complexion dimmed. And every fair from fair sometime declines. By chance or nature changing course untrimmed. But thy eternal summer shall not fade. Nor lose possession of that fair thou ow. Nor shall Death brag thou wand rest in his shade. When in eternal lines to time thou grow. So long as men can breathe or eyes can see. So long lives this,and this gives life to thee.'
for sentence in tokenizer.tokenize(sentences):
    sen1 += [nltk.word_tokenize(sentence)]

#for i in sen1:
    #sen2 += [word for word in i if word not in clean]

Vb = Vocab()
for i in sen1:
    Vb.fit(i)

Vb.build_vocab(min_count=1,max_word=13)
print(Vb.dict)
print(sen1)
#for i in sen1:
    #print(Vb.word2idx(i,max_len=13))

w2v = Word2Vec(sen1, vector_size=50, window=3, min_count=1)
print(w2v.wv.vectors.shape)
