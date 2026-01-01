import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import streamlit as st
import PyPDF2  # For extracting text from PDF

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForQuestionAnswering.from_pretrained("xlm-roberta-base")

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page].extract_text()
    return text

# Function to answer question
def answer_question(question, context):
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    return answer

# Streamlit UI for Q&A with PDF upload
def main():
    st.title("XLM-R Question Answering System with PDF Input")
    uploaded_file = st.file_uploader("Upload PDF file", type="pdf")
    
    if uploaded_file is not None:
        context = extract_text_from_pdf(uploaded_file)
        st.write("Extracted Text:", context[:2000])  # Display the first 2000 characters for debugging
        
        user_question = st.text_input("Enter your question:")
        
        if st.button("Get Answer"):
            if user_question and context:
                answer = answer_question(user_question, context)
                if answer.strip():
                    st.write("Answer:", answer)
                else:
                    st.write("No answer found.")
            else:
                st.write("Please upload a PDF file and enter a question.")

if __name__ == "__main__":
    main()

