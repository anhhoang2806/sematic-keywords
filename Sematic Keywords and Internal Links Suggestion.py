import streamlit as st
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Helper function to extract content from a URL
def extract_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs])
        return content
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching content from {url}: {e}")
        return ""

# Streamlit UI
def main():
    st.title("Semantic Keywords and Internal Links Suggestion Tool")

    # Input: List of URLs
    url_list = st.text_area("Enter URLs (one per line):").splitlines()
    url_list = [url.strip() for url in url_list if url.strip()]

    if st.button("Analyze"):
        if len(url_list) < 2:
            st.warning("Please enter at least two URLs to analyze similarity and suggest internal links.")
            return

        # Extract content from URLs
        st.info("Extracting content from URLs...")
        contents = [extract_content(url) for url in url_list]

        # Filter out URLs with empty content
        valid_urls_contents = [(url, content) for url, content in zip(url_list, contents) if content]
        if not valid_urls_contents:
            st.error("No valid content found from the provided URLs.")
            return

        valid_urls, valid_contents = zip(*valid_urls_contents)

        # Calculate TF-IDF and cosine similarity
        st.info("Calculating similarity between pages...")
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(valid_contents)
        except ValueError as e:
            st.error(f"Error calculating TF-IDF: {e}")
            return

        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Handle case where similarity matrix is empty or invalid
        if similarity_matrix.size == 0 or np.isnan(similarity_matrix).any():
            st.error("Error: Similarity calculation resulted in an invalid matrix.")
            return

        # Display similarity results
        st.subheader("Similarity Matrix")
        st.write(similarity_matrix)

        # Suggest internal links based on similarity
        st.subheader("Suggested Internal Links")
        for i in range(len(valid_urls)):
            similar_pages = [valid_urls[j] for j in range(len(valid_urls)) if i != j and similarity_matrix[i][j] > 0.1]
            if similar_pages:
                st.write(f"Pages similar to {valid_urls[i]}:")
                for page in similar_pages:
                    st.write(f"- {page}")

if __name__ == "__main__":
    main()
