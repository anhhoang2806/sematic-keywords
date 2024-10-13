import streamlit as st
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

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
    st.title("Semantic Keywords Suggestion Tool")

    # Input: List of URLs
    url_list = st.text_area("Enter URLs (one per line):").splitlines()
    url_list = [url.strip() for url in url_list if url.strip()]

    if st.button("Analyze"):
        if not url_list:
            st.warning("Please enter at least one URL to analyze.")
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

        # Calculate TF-IDF for semantic keywords
        st.info("Calculating semantic keywords...")
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(valid_contents)
        except ValueError as e:
            st.error(f"Error calculating TF-IDF: {e}")
            return

        feature_names = vectorizer.get_feature_names_out()

        # Display top semantic keywords for each URL
        st.subheader("Semantic Keywords for Each URL")
        for i, url in enumerate(valid_urls):
            st.write(f"Semantic Keywords for {url}:")
            tfidf_scores = zip(feature_names, tfidf_matrix[i].toarray()[0])
            sorted_keywords = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)[:10]
            for keyword, score in sorted_keywords:
                st.write(f"- {keyword}: {score:.4f}")

if __name__ == "__main__":
    main()
