# ir_system_nltk.py
import os
import string
import sys
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Set the documents folder relative to the script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(SCRIPT_DIR, "documents")

# NLTK utilities
STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# ---------- Utilities ----------
def read_documents(folder=DOCS_DIR):
    if not os.path.exists(folder):
        print(f"Error: The folder '{folder}' does not exist.")
        sys.exit(1)
    docs = {}
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(".txt"):
            path = os.path.join(folder, fname)
            with open(path, "r", encoding="utf-8") as f:
                docs[fname] = f.read()
    if not docs:
        print(f"No .txt files found in '{folder}'.")
        sys.exit(1)
    return docs

def tokenize(text):
    text = text.lower()
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    tokens = word_tokenize(text)
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens

# ---------- Index building ----------
def build_inverted_index(docs):
    inverted = {}  # term -> set(docnames)
    for docname, text in docs.items():
        tokens = set(tokenize(text))
        for t in tokens:
            inverted.setdefault(t, set()).add(docname)
    inverted_sorted = {t: sorted(list(docs_set)) for t, docs_set in inverted.items()}
    return inverted_sorted

def build_dictionary(inverted_index):
    return sorted(inverted_index.keys())

# ---------- Boolean query processing ----------
PREC = {"NOT": 3, "AND": 2, "OR": 1}

def tokenize_query(q):
    q = q.replace("(", " ( ").replace(")", " ) ")
    parts = q.split()
    tokens = []
    for p in parts:
        up = p.upper()
        if up in ("AND","OR","NOT","(",")"):
            tokens.append(up)
        else:
            # Use NLTK tokenizer & lemmatizer
            cleaned_tokens = [LEMMATIZER.lemmatize(t.lower()) 
                              for t in word_tokenize(p) 
                              if t.lower() not in STOPWORDS and len(t) > 1]
            tokens.extend(cleaned_tokens)
    return tokens

def infix_to_postfix(tokens):
    out = []
    stack = []
    for tok in tokens:
        if tok == "(":
            stack.append(tok)
        elif tok == ")":
            while stack and stack[-1] != "(":
                out.append(stack.pop())
            stack.pop()
        elif tok in PREC:
            while stack and stack[-1] != "(" and PREC.get(stack[-1],0) >= PREC[tok]:
                out.append(stack.pop())
            stack.append(tok)
        else:
            out.append(tok)
    while stack:
        out.append(stack.pop())
    return out

def eval_postfix(postfix, inverted_index, all_docs):
    stack = []
    all_docs_set = set(all_docs)
    for tok in postfix:
        if tok == "NOT":
            a = stack.pop()
            stack.append(all_docs_set - a)
        elif tok == "AND":
            b = stack.pop()
            a = stack.pop()
            stack.append(a & b)
        elif tok == "OR":
            b = stack.pop()
            a = stack.pop()
            stack.append(a | b)
        else:
            docs = set(inverted_index.get(tok, []))
            stack.append(docs)
    return stack[0]

def boolean_query(query_str, inverted_index, all_docs):
    tokens = tokenize_query(query_str)
    postfix = infix_to_postfix(tokens)
    result_set = eval_postfix(postfix, inverted_index, all_docs)
    return sorted(result_set), tokens, postfix

# ---------- CLI / demonstration ----------
def main():
    print("Reading documents from:", DOCS_DIR)
    docs = read_documents(DOCS_DIR)
    print(f"Loaded {len(docs)} documents:", list(docs.keys()))

    inverted = build_inverted_index(docs)
    dictionary = build_dictionary(inverted)

    print("\n=== Dictionary (first 40 terms) ===")
    print(dictionary[:40])
    print("\n=== Inverted index (first 20 terms) ===")
    for i, (term, doclist) in enumerate(sorted(inverted.items())[:20]):
        print(f"{term} -> {doclist}")

    print("\nEnter Boolean queries using AND, OR, NOT, parentheses. Examples:")
    print("  emily AND dickinson")
    print("  kafka OR fyodor")
    print("  (plath AND poetry) OR NOT fun")
    print("Type 'exit' to quit.\n")

    all_docs = list(docs.keys())
    while True:
        q = input("Query> ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            print("Exiting program.")
            break
        try:
            results, tokens, postfix = boolean_query(q, inverted, all_docs)
            print("Tokens:", tokens)
            print("Postfix:", postfix)
            print("Results ({} docs):".format(len(results)))
            for r in results:
                print(" -", r)
        except Exception as e:
            print("Error evaluating query:", e)

if __name__ == "__main__":
    main()
