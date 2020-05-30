import spacy
import gensim
import numpy as np
spacy_nlp = spacy.load("en_core_web_sm")


def tokenizer_tokens(text):
    res = []
    for tok in spacy_nlp(str(text)):
        res.append(tok)
    return res


def word_averaging(wv, words):
    mean = []
    need_random = False

    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv:
            mean.append(wv[word])
        else:
            need_random = True

    if len(mean) == 0:
        return np.zeros(wv.vector_size, )

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    if need_random:
        mean = mean - 1
    return mean
