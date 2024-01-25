import streamlit as st
import pickle

# Load the pickled vectorizer and model
with open('vectorizer.pickle', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open('svcModel(ok).pickle', 'rb') as model_file:
    sentiment_analysis_model = pickle.load(model_file)

# Streamlit app
def main():
    st.title("Review Classifier App")

    # User input
    user_input = st.text_area("Enter your review here:")

    # Button to trigger the review classification
    if st.button("Classify Review"):
        # Vectorize the user input
        user_input_vectorized = vectorizer.transform([user_input])

        # Perform sentiment analysis on the vectorized input
        result = sentiment_analysis_model.predict(user_input_vectorized)
        printer = ''
        if result[0] == 1:
            printer = 'positive'
        else:
            printer = 'negative'
        # Display the result
        st.header("Review Classification:")
        st.header(f"Review: {user_input}")
        st.header(f"Sentiment: {printer}")

if __name__ == "__main__":
    main()
