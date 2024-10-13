import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# Streamlit UI
def main():
    st.title("Semantic Keywords Suggestion Tool")

    # Input: Content
    content = st.text_area("Enter the content you want to analyze:")

    # Input: Optional keywords
    optional_keywords = st.text_area("Enter optional keywords (one per line):").splitlines()
    optional_keywords = [kw.strip() for kw in optional_keywords if kw.strip()]

    if st.button("Analyze"):
        if not content:
            st.warning("Please enter content to analyze.")
            return

        # Calculate TF-IDF for semantic keywords
        st.info("Calculating semantic keywords...")
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform([content])
        except ValueError as e:
            st.error(f"Error calculating TF-IDF: {e}")
            return

        feature_names = vectorizer.get_feature_names_out()

        # Display top semantic keywords for the content
        st.subheader("Semantic Keywords for the Content")
        tfidf_scores = zip(feature_names, tfidf_matrix.toarray()[0])
        sorted_keywords = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)

        # Display top 10 semantic keywords
        top_keywords = sorted_keywords[:10]
        for keyword, score in top_keywords:
            st.write(f"- {keyword}: {score:.4f}")

        # Check if optional keywords are in the content and their scores
        if optional_keywords:
            st.subheader("Optional Keywords Analysis")
            for kw in optional_keywords:
                if kw in feature_names:
                    index = feature_names.tolist().index(kw)
                    score = tfidf_matrix.toarray()[0][index]
                    st.write(f"- '{kw}' is present in the content with a score of {score:.4f}.")
                else:
                    st.write(f"- '{kw}' is NOT present in the content.")

if __name__ == "__main__":
    main()
