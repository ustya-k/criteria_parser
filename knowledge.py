import pickle
from owlready2.pymedtermino2.umls import *
VECTORS_PATH = "./data/snomedct_weights_wo_stacking.pickle"

SNOMED_PATH = "./data/umls-2019AB-metathesaurus.zip"

default_world.set_backend(filename = "pym.sqlite3")
import_umls(SNOMED_PATH, terminologies=["SNOMEDCT_US", "CUI", "NCI"])
default_world.save()


class KnowledgeBase:

    def __init__(self):
        self.PYM = get_ontology("http://PYM/").load()
        self.SNOMEDCT = self.get_snomed()
        self.CUI = self.get_cui()
        self.NCI = self.get_nci()
        with open(VECTORS_PATH, "rb") as f:
            self.vectors = pickle.load(f)
        self.cached_terms = dict()

    def get_snomed(self):
        return self.PYM["SNOMEDCT_US"]

    def get_cui(self):
        return self.PYM["CUI"]

    def get_nci(self):
        return self.PYM["NCI"]

    def search_snomed(self, id_=None, text=None):
        if id_ is not None:
            if self.SNOMEDCT.has_concept(id_):
                return self.SNOMEDCT[id_]
            return None
        text = text.replace("-", " ")
        text = text.replace("$", " ")
        return self.SNOMEDCT.search(text)

    def search_cui(self, id_=None, text=None):
        if id_ is not None:
            if self.CUI.has_concept(id_):
                return self.CUI[id_]
            return None
        text = text.replace("-", " ")
        return self.CUI.search(text)

    def search_nci(self, id_=None, text=None):
        if id_ is not None:
            if self.NCI.has_concept(id_):
                return self.NCI[id_]
            return None
        text = text.replace("-", " ")
        return self.NCI.search(text)
