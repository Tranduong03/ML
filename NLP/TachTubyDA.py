import re
import math
from collections import defaultdict


def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    return tokens

def load_corpus(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    tokens = clean_and_tokenize(raw_text)
    return raw_text.splitlines(), tokens

# lines, corpus_tokens = load_corpus("english_sample.txt")
lines, corpus_tokens = load_corpus(r"D:/IT/HK1_Y4/Class/ML/NLP/english_sample.txt")



eng_dict = set(corpus_tokens)

def build_unigram(corpus_tokens):
    counts = defaultdict(int)
    total = 0
    for t in corpus_tokens:
        counts[t]+=1
        total +=1
    probs = {w: counts[w]/total for w in counts}
    return probs

probs = build_unigram(corpus_tokens)


def fmm(s, dictionary, max_len=None):
    if max_len is None:
        max_len = max((len(w) for w in dictionary), default=len(s))
    i, n = 0, len(s)
    tokens = []
    while i < n:
        matched = False
        for L in range(min(max_len, n-i), 0, -1):
            piece = s[i:i+L]
            if piece in dictionary:
                tokens.append(piece)
                i += L
                matched = True
                break
        if not matched:
            tokens.append(s[i])
            i += 1
    return tokens


def bmm(s, dictionary, max_len=None):
    if max_len is None:
        max_len = max((len(w) for w in dictionary), default=len(s))
    tokens = []
    i = len(s)
    while i > 0:
        matched = False
        for L in range(min(max_len, i), 0, -1):
            piece = s[i-L:i]
            if piece in dictionary:
                tokens.insert(0, piece)
                i -= L
                matched = True
                break
        if not matched:
            tokens.insert(0, s[i-1])
            i -= 1
    return tokens


def viterbi_unigram(s, probs, unk=1e-8, max_word_len=20):
    n = len(s)
    best = [-1e9]*(n+1)
    prev = [-1]*(n+1)
    best[0] = 0.0
    for i in range(n):
        if best[i] < -1e8: continue
        for j in range(i+1, min(n, i+max_word_len)+1):
            w = s[i:j]
            p = probs.get(w, unk)
            score = best[i] + math.log(p)
            if score > best[j]:
                best[j] = score
                prev[j] = i
    if prev[n] == -1:
        return [s]
    toks = []
    idx = n
    while idx > 0:
        toks.append(s[prev[idx]:idx])
        idx = prev[idx]
    toks.reverse()
    return toks


for line in lines:
    clean_line = re.sub(r"[^a-z0-9]", "", line.lower())
    if not clean_line:
        continue
    print(f"\nSentence: {line.strip()}")
    print("Input (no space):", clean_line)
    print("FMM:", fmm(clean_line, eng_dict))
    print("BMM:", bmm(clean_line, eng_dict))
    print("Viterbi:", viterbi_unigram(clean_line, probs))
