import fitz  # PyMuPDF
import re
import spacy
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize the QA and summarization models
qa_model = pipeline("question-answering")
summarizer = pipeline("summarization")

def extract_text_from_pdf(pdf_file):
    text = ""
    # Use fitz.open with the file-like object
    with fitz.open("pdf", pdf_file.read()) as doc:
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
    return text

def clean_text(text):
    # Remove multiple spaces and hyphenated word splits
    cleaned_text = re.sub(r'-\s+', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
   
    # Remove leading/trailing spaces
    cleaned_text = cleaned_text.strip()
   
    return cleaned_text

def get_answer_from_qa_model(text, question):
    result = qa_model(question=question, context=text)
    return result['answer']

def summarize_text(text, max_length=150):
    summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def find_relevant_sentences(text, question, top_n=5):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
   
    # Vectorize the sentences and the question
    vectorizer = TfidfVectorizer().fit([question] + sentences)
    question_vec = vectorizer.transform([question])
    sentence_vecs = vectorizer.transform(sentences)
   
    # Compute cosine similarity
    cosine_similarities = cosine_similarity(question_vec, sentence_vecs).flatten()
   
    # Sort sentences by similarity score
    top_sentence_indices = cosine_similarities.argsort()[-top_n:][::-1]
    top_sentences = [sentences[i] for i in top_sentence_indices]
   
    # Aggregate the top sentences to form the answer
    aggregated_answer = " ".join(top_sentences)
   
    # Use the QA model to get a precise answer
    precise_answer = get_answer_from_qa_model(aggregated_answer, question)
   
    # Summarize the precise answer
    summarized_answer = summarize_text(precise_answer)
   
    return {"question": question, "answer": summarized_answer}

def main():
    st.title("PDF Question-Answer System")

    pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

    if pdf_file is not None:
        raw_text = extract_text_from_pdf(pdf_file)
        cleaned_text = clean_text(raw_text)
       
        st.write("## Enter your question:")
        question = st.text_input("Question")
       
        if st.button("Get Answer"):
            if question:
                answer = find_relevant_sentences(cleaned_text, question)
                st.write("## Answer:")
                st.write(f"**Question:** {answer['question']}")
                st.write(f"**Answer:** {answer['answer']}")
                st.write("---")
            else:
                st.write("Please enter a question.")
   
if __name__ == "__main__":
    main()
