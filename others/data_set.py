import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer, T5Tokenizer, T5ForConditionalGeneration
import torch
import PyPDF2

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page].extract_text()
    return text

# Function for text summarization using BART
def summarize_text_bart(text):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = model.generate(inputs['input_ids'], max_length=1000, min_length=400, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function for question answering using T5
def answer_question_t5(question, text):
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')

    # Adjust the question format according to T5 requirements
    input_text = "question: " + question + " context: " + text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate answer
    answer_ids = model.generate(inputs.input_ids, max_length=32, num_beams=4, early_stopping=True)
    answer = tokenizer.decode(answer_ids[0], skip_special_tokens=True)

    return answer

# Initialize Streamlit app
st.set_page_config(layout="wide")
st.title("PDF Text Summarization and Question Answering with BART and T5")

# Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Extract text from PDF
    pdf_text = extract_text_from_pdf(uploaded_file)

    # Summarize the extracted text using BART
    summarized_text = summarize_text_bart(pdf_text)

    # Display summarized text
    st.header("Summarized Text:")
    st.write(summarized_text)

    # Ask a question about the original text
    user_question = st.text_input("Enter your question about the original text:")
    if st.button("Get Answer"):
        if user_question:
            answer = answer_question_t5(user_question, pdf_text)
            st.header("Answer:")
            st.write(answer)
        else:
            st.warning("Please enter a question.")