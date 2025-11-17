import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


folder_path = "documents/"


documents = []
filenames = []

for file in os.listdir(folder_path):
    if file.endswith(".txt"):   # only read txt files
        with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
            documents.append(f.read())
            filenames.append(file)


print("Loaded documents:", filenames)

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

similarity_matrix = cosine_similarity(tfidf_matrix)


print("\nDocument Similarity Matrix:\n")
print(similarity_matrix)

print("\nSimilarity Scores Between Documents:\n")
for i, file1 in enumerate(filenames):
    for j, file2 in enumerate(filenames):
        print(f"{file1}  <->  {file2}  =  {similarity_matrix[i][j]:.4f}")
    print()
