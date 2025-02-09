# tfidf_keyword_extractor.py

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def extract_keywords_tfidf(job_description, num_keywords=10):
    """
    Extract keywords from a job description using TF-IDF.
    
    Args:
        job_description (str): The job description text.
        num_keywords (int): Number of top keywords to return.
    
    Returns:
        List of tuples: Each tuple contains (word, score).
    """
    # In this simple case, we are using a single document.
    # For a better TF-IDF model, you'd normally have a corpus of documents.
    documents = [job_description]

    # Initialize TfidfVectorizer with English stopwords
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Get feature names and their TF-IDF scores
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]

    # Sort the words by their score in descending order
    sorted_indices = np.argsort(scores)[::-1]
    top_keywords = [(feature_names[i], scores[i]) for i in sorted_indices if scores[i] > 0][:num_keywords]

    return top_keywords

if __name__ == "__main__":
    job_description = """
    We are seeking a software engineer with expertise in Python, Java, and cloud technologies.
    The candidate should have experience with RESTful APIs, microservices architecture, and containerization
    using Docker and Kubernetes. Strong communication skills and a collaborative mindset are essential.
    Experience in agile methodologies and familiarity with CI/CD pipelines is a plus.
    """

    keywords = extract_keywords_tfidf(job_description, num_keywords=10)
    
    print("Top keywords from the job description (TF-IDF):")
    for word, score in keywords:
        print(f"Keyword: {word} - Score: {score:.4f}")
