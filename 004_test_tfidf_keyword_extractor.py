import os
import glob
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def load_corpus_from_directory(directory):
    """
    Load all .txt files from the given directory and return a list of their contents.

    Args:
        directory (str): Path to the directory containing text files.

    Returns:
        list of str: List of file contents (each representing a job description).
    """
    corpus = []
    # Search for all .txt files in the specified directory.
    for filepath in glob.glob(os.path.join(directory, "*.txt")):
        with open(filepath, 'r', encoding='utf-8') as file:
            corpus.append(file.read())
    return corpus

def train_tfidf_vectorizer(corpus):
    """
    Train a TfidfVectorizer on a given corpus of job descriptions.

    Args:
        corpus (list of str): List of job descriptions.

    Returns:
        TfidfVectorizer: A vectorizer fitted on the corpus.
    """
    # Initialize the vectorizer with English stopwords.
    vectorizer = TfidfVectorizer(stop_words='english')
    # Fit the vectorizer on the corpus to build the vocabulary and compute IDF values.
    vectorizer.fit(corpus)
    return vectorizer

def extract_keywords_tfidf(new_job_description, vectorizer, num_keywords=40):
    """
    Extract keywords from a new job description using a pre-trained TF-IDF vectorizer.
    
    Args:
        new_job_description (str): The job description text.
        vectorizer (TfidfVectorizer): A TF-IDF vectorizer that has been fitted on a corpus.
        num_keywords (int): Number of top keywords to return.
    
    Returns:
        list of tuples: Each tuple contains (word, tf-idf score).
    """
    # Transform the new job description using the fitted vectorizer.
    tfidf_vector = vectorizer.transform([new_job_description])
    # Get the vocabulary terms (feature names).
    feature_names = vectorizer.get_feature_names_out()
    # Convert the sparse TF-IDF vector to a dense array.
    dense_vector = tfidf_vector.toarray()[0]
    # Get indices of terms sorted by their TF-IDF score in descending order.
    sorted_indices = np.argsort(dense_vector)[::-1]
    # Extract the top keywords with non-zero scores.
    top_keywords = [
        (feature_names[i], dense_vector[i])
        for i in sorted_indices if dense_vector[i] > 0
    ][:num_keywords]
    return top_keywords

if __name__ == '__main__':
    # Define the directory that contains your corpus of job descriptions as .txt files.
    corpus_directory = 'corpus'
    corpus = load_corpus_from_directory(corpus_directory)

    if not corpus:
        print(f"No text files found in the directory: {corpus_directory}")
        exit(1)

    # Train the TF-IDF vectorizer using the corpus.
    vectorizer = train_tfidf_vectorizer(corpus)

    # Read the new job description from a separate text file.
    job_description_file = 'job_description.txt'
    try:
        with open(job_description_file, 'r', encoding='utf-8') as file:
            new_job_description = file.read()
    except FileNotFoundError:
        print(f"The file '{job_description_file}' was not found. Please check the file path.")
        exit(1)

    # Extract and print the top keywords from the new job description.
    keywords = extract_keywords_tfidf(new_job_description, vectorizer, num_keywords=30)

    print("Top keywords from the job description (TF-IDF):")
    for word, score in keywords:
        print(f"Keyword: {word} - Score: {score:.4f}")
