from .word import Word
from .knowledge import KnowledgeBase
from .helpers import *
from gensim import models
from scipy import spatial
import spacy
nlp = spacy.load("en_core_web_sm")

KNOWLEDGE = KnowledgeBase()
W2VMODEL = models.KeyedVectors.load_word2vec_format("./data/wikipedia-pubmed-and-PMC-w2v.bin", binary=True)


class Criterion:

    def __init__(self, text):
        self.text = text
        self.doc = nlp(self.text)
        self.root = None
        self.words = set()
        self.build_tree()
        self.json = {"text": text, "tree": []}
        self.numeric = list()
        self.terms = None

    def _find_head(self):
        return [el for el in self.doc if el.dep_ == "ROOT"][0]

    def build_tree(self):
        w = self._find_head()
        self.root = Word(w.text, int(w.i))
        self.root.params = w
        self.words = {int(w.i): self.root}
        queue = list(w.children)
        while len(queue) != 0:
            w = queue.pop(0)
            curr = Word(w.text, int(w.i))
            curr.params = w
            curr_new_rel = {"relation": w.dep_, "word": self.words[w.head.i], "direction": -1}
            curr.relations.append(curr_new_rel)
            head_new_rel = {"relation": w.dep_, "word": curr, "direction": 1}
            self.words[w.head.i].relations.append(head_new_rel)
            self.words[curr.id] = curr
            queue += list(w.children)

    def distance(self, text1, text2):
        tokenized1 = [tok.text.lower() for tok in tokenizer_tokens(text1)]
        vect1 = word_averaging(W2VMODEL, tokenized1)

        tokenized2 = [tok.text.lower() for tok in tokenizer_tokens(text2)]
        vect2 = word_averaging(W2VMODEL, tokenized2)
        return spatial.distance.cosine(vect1, vect2)


    def check_full_match(self, text, lemmatized=None, source="ALL"):
        if text in KNOWLEDGE.cached_terms:
            return KNOWLEDGE.cached_terms[text]
        tokenized = tokenizer_tokens(text)
        textlower = " ".join([tok.text.lower() for tok in tokenized])
        if lemmatized is not None:
            textlowerlemmas = lemmatized
        else:
            textlowerlemmas = " ".join([tok.lemma_.lower() for tok in tokenized])
        if source == "SNOMED":
            res = KNOWLEDGE.search_snomed(text=text)
        elif source == "CUI":
            res = KNOWLEDGE.search_cui(text=text)
        elif source == "NCI":
            res = KNOWLEDGE.search_nci(text=text)
        elif source == "ALL":
            res = set(KNOWLEDGE.search_snomed(text=textlower))
            res.update(set(KNOWLEDGE.search_nci(text=textlower)))

            if textlower != textlowerlemmas:
                res.update(set(KNOWLEDGE.search_snomed(text=textlowerlemmas)))
                res.update(set(KNOWLEDGE.search_nci(text=textlowerlemmas)))

        probable = []
        probable_lemma = []
        res = {el for el in res if re.search("^(C\d+|\d+)$", el.name)}
        for concept in res:
            try:
                min_dist = 1
                min_dist_lemmas = 1
                for syn in concept.synonyms + concept.label:
                    d = self.distance(syn, text)
                    dlemmas = self.distance(syn, textlowerlemmas)
                    if d < min_dist:
                        min_dist = d
                    if dlemmas < min_dist_lemmas:
                        min_dist_lemmas = dlemmas
                probable.append((concept, min_dist))
                probable_lemma.append((concept, min_dist_lemmas))
            except:
                continue
        probable.sort(key=lambda l: l[1])
        if len(probable) != 0 and probable[0][1] < 0.05:
            return probable[0][0]
        probable_lemma.sort(key=lambda l: l[1])
        if len(probable_lemma) != 0 and probable_lemma[0][1] < 0.05:
            return probable_lemma[0][0]
        return None

    def find_closest_term_child(self, concept, text, depth=1):
        names = concept.label + concept.synonyms
        texts = [text + " " + el for el in names]
        children = []
        check = set(concept.children)
        if hasattr(concept, "has_associated_morphology"):
            check.update(set(concept.has_associated_morphology))
        if hasattr(concept, "associated_morphology_of"):
            check.update(set(concept.associated_morphology_of))
        for t in texts:
            try:
                f = self.check_full_match(t)
                KNOWLEDGE.cached_terms[t] = f
                if f is not None and (f in check or concept in f.ancestors()):
                    min_dist = 1
                    labels = f.synonyms + f.label
                    for name in labels:
                        d = self.distance(name, t)
                        if d < min_dist:
                            min_dist = d
                    children.append((f, min_dist))
            except:
                print(t)
        children.sort(key=lambda l: l[1])
        if len(children) == 0:
            return None
        if children[0][1] < 0.05:
            return children[0][0]

        if depth == 2:
            res = []
            for child in concept.children:
                t = self.find_closest_term_child(child, text, depth=1)
                if t is not None:
                    res.append(t)
            res.sort(key=lambda l: l[1])
            if len(res) == 0:
                return None
            return res[0]

        return None

    def process_children(self, children):
        relations = []
        for el in children:
            res = self.look_up(el["word"])
            res["relation"] = el["relation"]
            relations.append(res)
        return relations

    def look_up(self, node):
        if node.params.pos_ not in ["ADJ", "ADV", "NOUN", "PROPN", "VERB"]:
            node_out_relations = [el for el in node.relations if el["direction"] == 1]
            concepts = {"text": node.text, "term": None, "cui": None, "relations": self.process_children(node_out_relations)}
            return concepts
        possible_term_parts = [el for el in node.relations if el["direction"] == 1
                    and el["word"].params.pos_ in ["ADJ", "ADV", "NOUN", "PROPN", "VERB"]
                    and el["relation"] != "conj" and el["relation"] != "compound" and node.id > el["word"].id]
        other_node_relations = [el for el in node.relations if el["direction"] == 1
                          and (node.id < el["word"].id or el["relation"] == "conj" or
                               el["word"].params.pos_ not in ["ADJ", "ADV", "NOUN", "PROPN", "VERB"])
                          and el["relation"] != "compound"]
        compound = [node] + [el["word"] for el in node.relations if el["relation"] == "compound"]
        compound.sort(key=lambda l: l.id)
        text = " ".join([el.text for el in compound])
        lemmatized = " ".join([el.params.lemma_ for el in compound])
        umls_search_result = self.check_full_match(text, lemmatized=lemmatized)
        KNOWLEDGE.cached_terms[text] = umls_search_result
        concepts = {"relation": "root"}

        possible_term_parts.sort(key=lambda k: abs(node.id - k["word"].id))

        if umls_search_result is None:
            concepts = {"text": text, "term": None, "cui": None}

        check = True
        i = 0
        curr = text
        res = umls_search_result
        prev = None
        while check and i < len(possible_term_parts) and res is not None:
            prev = curr
            curr = possible_term_parts[i]["word"].text + " " + curr
            closest = self.find_closest_term_child(res, possible_term_parts[i]["word"].text)
            if closest is None:
                check = False
            else:
                i += 1
                res = closest

        if res is not None:
            concepts["term"] = res.label[0]
            concepts["cui"] = (res >> KNOWLEDGE.CUI).pop().name

        if prev is None or check:
            concepts["text"] = curr
        else:
            concepts["text"] = prev

        to_process = other_node_relations + possible_term_parts[i:]
        concepts["relations"] = self.process_children(to_process)
        return concepts

    def detect_terms(self):
        self.json["tree"] = self.look_up(self.root)
