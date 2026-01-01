import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer, MarianMTModel, MarianTokenizer
import PyPDF2
from gtts import gTTS
import tempfile

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

# Function to convert text to audio using gTTS
def text_to_audio(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
        tts.save(temp_audio_file.name)
        return temp_audio_file.name

# Function to translate text
def translate_text(text, src_lang, tgt_lang):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    translated_ids = model.generate(inputs["input_ids"], max_length=512, num_beams=4, early_stopping=True)
    translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    return translated_text

# Initialize Streamlit app
st.set_page_config(layout="wide")
st.title("PDF Text Summarization with BART")

# Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Extract text from PDF
    pdf_text = extract_text_from_pdf(uploaded_file)

    # Summarize the extracted text using BART
    summarized_text = summarize_text_bart(pdf_text)

    # Display summarized text
    st.write("Summarized Text:")
    st.write(summarized_text)

    # Convert summarized text to audio
    audio_file = text_to_audio(summarized_text)

    # Button to listen to the audio summary
    if st.button("Listen to Summarized Text"):
        audio_bytes = open(audio_file, "rb").read()
        st.audio(audio_bytes, format="audio/mp3")

    # Translate the summarized text
    st.write("## Translate Summary:")
    tgt_lang = st.selectbox("Select target language", ["es", "fr", "de", "it", "pt", "zh"])
    if st.button("Translate Summary"):
        translated_summary = translate_text(summarized_text, "en", tgt_lang)
        st.write("## Translated Summary:")
        st.write(translated_summary)
        if st.button("Listen to Translated Summary"):
            audio_file_translated_summary = text_to_audio(translated_summary)
            audio_bytes_translated_summary = open(audio_file_translated_summary, "rb").read()
            st.audio(audio_bytes_translated_summary, format="audio/mp3")
