import streamlit as st
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Helper function to extract content from a URL
def extract_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs])
        return content
    except Exception as e:
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

        # Calculate TF-IDF and cosine similarity
        st.info("Calculating similarity between pages...")
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(contents)
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Display similarity results
        st.subheader("Similarity Matrix")
        st.write(similarity_matrix)

        # Suggest internal links based on similarity
        st.subheader("Suggested Internal Links")
        for i in range(len(url_list)):
            similar_pages = [url_list[j] for j in range(len(url_list)) if i != j and similarity_matrix[i][j] > 0.1]
            if similar_pages:
                st.write(f"Pages similar to {url_list[i]}:")
                for page in similar_pages:
                    st.write(f"- {page}")

if __name__ == "__main__":
    main()