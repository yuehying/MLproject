import preprop as preprop
import logging
from gensim.models import Word2Vec
import numpy as np
import analysis

# control flags
RE_TRAIN_EMBBED = False

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

file = open('../small_dtrain.csv','r')
file = preprop.decoding(file)
y = preprop.getLabel(file)


# generate
uni_gram = preprop.genNgram(file, 1)
bi_gram = preprop.genNgram(file, 2)
tri_gram = preprop.genNgram(file, 3)

if RE_TRAIN_EMBBED:
    w2v_uni = Word2Vec(sentences=uni_gram, size=10, window=5, min_count=1, workers=2,
                 sg=1, iter=10)
    w2v_uni.save("w2v_uni.model")
    w2v_bi = Word2Vec(sentences=bi_gram, size=100, window=5, min_count=1, workers=2,
                       sg=1, iter=10)
    w2v_bi.save("w2v_bi.model")
    #w2v_tri = Word2Vec(sentences=tri_gram, size=100, window=5, min_count=1, workers=2,
    #                   sg=1, iter=10)
    #w2v_tri.save("w2v_tri.model")

w2v_uni = Word2Vec.load("w2v_uni.model")
w2v_bi = Word2Vec.load("w2v_bi.model")
#w2v_tri = Word2Vec.load("w2v_tri.model")


bi_gram_keys = set()
for line in uni_gram:
    for key in line:
        bi_gram_keys.add(key)
n_gram_keys = sorted(list(bi_gram_keys))
n_gram_keys = [(key, idx) for (idx,key) in enumerate(n_gram_keys)]
n_gram_keys = dict(n_gram_keys)
val = len(n_gram_keys.keys())

# use word2vec model to obtain each embedded key
X = np.array([w2v_uni.wv[key] for key in n_gram_keys])
analysis.t_SNE_visualization(X, n_gram_keys)

print(10)