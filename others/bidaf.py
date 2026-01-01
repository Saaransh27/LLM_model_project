import streamlit as st
from allennlp.predictors.predictor import Predictor
import allennlp_models.rc
import pdfplumber

# Load pre-trained BiDAF model and predictor
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bidaf-model-2020.03.19.tar.gz")

def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

def answer_question(question, context):
    # Use BiDAF model to answer the question
    prediction = predictor.predict(passage=context, question=question)
    answer = prediction['best_span_str']
    return answer

def main():
    st.title("PDF Question Answering with BiDAF")

    # File upload
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        question = st.text_input("Ask a question")

        # Process PDF file
        text = extract_text_from_pdf(uploaded_file)

        # Show original text
        st.subheader("Original Text")
        st.text(text[:5000])  # Display a snippet of the text

        # Answer questions
        if question:
            st.subheader("Question:")
            st.text(question)
            st.subheader("Answer:")
            answer = answer_question(question, text)
            st.text(answer)

if __name__ == "__main__":
    main()