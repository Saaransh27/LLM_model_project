import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer, BertTokenizer, BertForQuestionAnswering
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

# Function for question answering using BERT
def answer_question_bert(question, text):
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="pt")
    start_positions, end_positions = model(**inputs)

    answer_start = torch.argmax(start_positions)
    answer_end = torch.argmax(end_positions) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    
    return answer

# Initialize Streamlit app
st.set_page_config(layout="wide")
st.title("PDF Text Summarization and Question Answering with BART and BERT")

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
    user_question = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        if user_question:
            answer = answer_question_bert(user_question, pdf_text)
            st.header("Answer:")
            st.write(answer)
        else:
            st.warning("Please enter a question.")

