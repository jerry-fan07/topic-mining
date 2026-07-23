from pathlib import Path
import numpy as np
import re
from NMF import NMF
import scipy.sparse as sp


def read_text(path):
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-16")


def vocabulary():
    doc_dict = {} # word to set of doc names
    num_dict = {} # num to word (total vocab)
    doc_counts = {} # doc to dict[word to count in doc]
    total_counts = {} # doc to total count
    for t in Path("./text").iterdir():
        dc = doc_counts[t.stem] = {}
        words = re.findall(r"\w+", read_text(t).lower())
        total_counts[t.stem] = len(words)
        for word in words:
            if word not in doc_dict:  # first time we see this word -> assign it an index
                num_dict[len(num_dict)] = word
            if word not in doc_dict:
                doc_dict[word] = {t.stem}
            else:
                doc_dict[word].add(t.stem)
            if word not in dc:
                dc[word] = 1
            else:
                dc[word] += 1

    return doc_dict, num_dict, doc_counts, total_counts

def word_document_matrix():
    doc_dict, num_dict, doc_counts, total_counts = vocabulary()
    words = [num_dict[i] for i in range(len(num_dict))]  # row order of the matrix
    matrix = np.array([[doc_counts[d][num_dict[i]] / total_counts[d] if num_dict[i] in doc_counts[d] else 0
                        for d in list(total_counts)]
                       for i in range(len(num_dict))])
    return matrix, words

def run(clusters: int):
    matrix, words = word_document_matrix()
    W, H = NMF(sp.csr_array(matrix), clusters, 10)
    return W, H, words
