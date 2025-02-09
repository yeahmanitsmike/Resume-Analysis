from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def train_tfidf_vectorizer(corpus):
    """
    Train a TfidfVectorizer on a given corpus of job descriptions.

    Args:
        corpus (list of str): List of job descriptions.

    Returns:
        TfidfVectorizer: A vectorizer fitted on the corpus.
    """
    # Initialize the vectorizer with English stopwords
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Fit the vectorizer on the corpus (this builds the vocabulary and computes IDF values)
    vectorizer.fit(corpus)
    
    return vectorizer

def extract_keywords_tfidf(new_job_description, vectorizer, num_keywords=20):
    """
    Extract keywords from a new job description using a pre-trained TF-IDF vectorizer.
    
    Args:
        new_job_description (str): The job description text.
        vectorizer (TfidfVectorizer): A TF-IDF vectorizer that has been fitted on a corpus.
        num_keywords (int): Number of top keywords to return.
    
    Returns:
        list of tuples: Each tuple contains (word, tf-idf score).
    """
    # Transform the new job description using the fitted vectorizer
    tfidf_vector = vectorizer.transform([new_job_description])
    
    # Get the vocabulary terms (feature names)
    feature_names = vectorizer.get_feature_names_out()
    
    # Convert the sparse TF-IDF vector to a dense array
    dense_vector = tfidf_vector.toarray()[0]
    
    # Sort indices of terms by their TF-IDF score in descending order
    sorted_indices = np.argsort(dense_vector)[::-1]
    
    # Extract the top keywords with non-zero scores
    top_keywords = [
        (feature_names[i], dense_vector[i])
        for i in sorted_indices if dense_vector[i] > 0
    ][:num_keywords]
    
    return top_keywords

if __name__ == '__main__':
    # Step 1: Define a sample corpus of job descriptions.
    corpus = [
        "We are seeking a software engineer with expertise in Python, Java, and cloud technologies.",
        "The candidate should have experience with RESTful APIs, microservices, and containerization using Docker.",
        "Join our dynamic team as a data scientist with strong skills in machine learning, data mining, and statistics.",
        "Looking for a project manager experienced in agile methodologies, budgeting, and team leadership.",
        "We are hiring a DevOps engineer proficient in AWS, Kubernetes, and CI/CD pipelines.",
        "An opportunity for a full-stack developer skilled in JavaScript, React, Node.js, and database management.",
        "Seeking a cybersecurity specialist with experience in threat detection, risk analysis, and incident response."
    ]
    
    # Step 2: Train the TF-IDF vectorizer on the corpus.
    vectorizer = train_tfidf_vectorizer(corpus)
    
    # Step 3: Read the job description from a text file.
    # Make sure to have a file named 'job_description.txt' in the same directory.
    try:
        with open('job_description.txt', 'r', encoding='utf-8') as file:
            new_job_description = file.read()
    except FileNotFoundError:
        print("The file 'job_description.txt' was not found. Please check the file path.")
        exit(1)
    
    # Step 4: Extract and print the top keywords based on their TF-IDF scores.
    keywords = extract_keywords_tfidf(new_job_description, vectorizer, num_keywords=10)
    
    print("Top keywords from the job description (TF-IDF):")
    for word, score in keywords:
        print(f"Keyword: {word} - Score: {score:.4f}")
