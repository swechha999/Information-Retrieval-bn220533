import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Path to your documents folder
folder_path = "documents/"

# Read all .txt files from the folder
documents = []
filenames = []

for file in os.listdir(folder_path):
    if file.endswith(".txt"):   # only read txt files
        with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
            documents.append(f.read())
            filenames.append(file)

# Check if documents loaded
print("Loaded documents:", filenames)

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(tfidf_matrix)

# Display results
print("\nDocument Similarity Matrix:\n")
print(similarity_matrix)

# Print matrix with labels for clarity
print("\nSimilarity Scores Between Documents:\n")
for i, file1 in enumerate(filenames):
    for j, file2 in enumerate(filenames):
        print(f"{file1}  <->  {file2}  =  {similarity_matrix[i][j]:.4f}")
    print()
