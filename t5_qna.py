import fitz  # PyMuPDF
import re
import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer, MarianMTModel, MarianTokenizer
import spacy
from PIL import Image
import ocr_module
from gtts import gTTS
import tempfile
import speech_recognition as sr
import sounddevice as sd
import scipy.io.wavfile as wav

# Load T5 model and tokenizer
model_name = "t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

nlp = spacy.load("en_core_web_sm")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open("pdf", pdf_file.read()) as doc:
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
    return text

# Function to clean text
def clean_text(text):
    cleaned_text = re.sub(r'-\s+', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = cleaned_text.strip()
    return cleaned_text

# Function to summarize text
def summarize_text(text):
    input_text = f"summarize: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs.input_ids, max_length=150, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# Function to generate answer
def generate_answer(text, question):
    input_text = f"question: {question} context: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs.input_ids, max_length=150, num_beams=4, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Function to refine answer
def refine_answer(answer, question):
    doc = nlp(answer)
    key_phrases = [chunk.text for chunk in doc.noun_chunks if chunk.text.lower() in question.lower()]
    entities = [ent.text for ent in doc.ents if ent.text.lower() in question.lower()]
    refined_answer = ' '.join(set(key_phrases + entities))
    return refined_answer if refined_answer else answer

# Function to convert text to speech
def text_to_speech(text):
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

# Function to record audio
def record_audio(duration, sample_rate=44100):
    st.write("Recording... Please speak into the microphone.")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    st.write("Recording finished.")
    return recording, sample_rate

# Function to save WAV file
def save_wav(audio_data, sample_rate, file_path):
    wav.write(file_path, sample_rate, audio_data)

# Function to convert speech to text
def speech_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Sorry, I could not understand the audio."
        except sr.RequestError:
            return "Sorry, there was an issue with the speech recognition service."

def main():
    st.set_page_config(page_title="Multi-format Input QA System", page_icon=":globe_with_meridians:")
    st.title("Multi-format Input Question-Answer System")

    if "recording_started" not in st.session_state:
        st.session_state.recording_started = False

    if "cleaned_text" not in st.session_state:
        st.session_state.cleaned_text = ""

    if "question" not in st.session_state:
        st.session_state.question = ""

    if "answer" not in st.session_state:
        st.session_state.answer = ""

    if "audio_file" not in st.session_state:
        st.session_state.audio_file = ""

    if "translated_summary" not in st.session_state:
        st.session_state.translated_summary = ""

    if "translated_answer" not in st.session_state:
        st.session_state.translated_answer = ""

    option = st.selectbox(
        "Select input format",
        ("Select an option", "Image", "PDF", "Text")
    )

    if option == "Image":
        image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if image_file:
            image = Image.open(image_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            extracted_text = ocr_module.detect_text(image)
            cleaned_text = clean_text(extracted_text)
            st.session_state.cleaned_text = cleaned_text

            if st.checkbox("View extracted text"):
                st.write("## Detected Text from Image:")
                st.write(cleaned_text)

    elif option == "PDF":
        pdf_file = st.file_uploader("Upload a PDF file", type="pdf")
        if pdf_file:
            raw_text = extract_text_from_pdf(pdf_file)
            cleaned_text = clean_text(raw_text)
            st.session_state.cleaned_text = cleaned_text

            if st.checkbox("View extracted text"):
                st.write("## Extracted Text from PDF:")
                st.write(cleaned_text)

    elif option == "Text":
        input_text = st.text_area("Enter your text")
        if input_text:
            cleaned_text = clean_text(input_text)
            st.write("## Input Text:")
            st.write(cleaned_text)
            st.session_state.cleaned_text = cleaned_text

    if option in ["Image", "PDF", "Text"] and st.session_state.cleaned_text:
        if st.checkbox("Summarize the content"):
            summary = summarize_text(st.session_state.cleaned_text)
            st.write("## Summary:")
            st.write(summary)
            if st.button("Listen to the summary"):
                audio_file = text_to_speech(summary)
                audio_bytes = open(audio_file, "rb").read()
                st.audio(audio_bytes, format="audio/mp3")

            st.write("## Translate Summary:")
            tgt_lang = st.selectbox("Select target language", ["es", "fr", "de", "it", "pt", "zh"])
            if st.button("Translate Summary"):
                translated_summary = translate_text(summary, "en", tgt_lang)
                st.session_state.translated_summary = translated_summary
                st.write("## Translated Summary:")
                st.write(translated_summary)
                if st.button("Listen to Translated Summary"):
                    audio_file_translated_summary = text_to_speech(translated_summary)
                    audio_bytes_translated_summary = open(audio_file_translated_summary, "rb").read()
                    st.audio(audio_bytes_translated_summary, format="audio/mp3")

        st.write("## Ask a question about the content:")
        question = st.text_input("Question", value=st.session_state.question)

        if st.button("Start Recording Question"):
            st.session_state.recording_started = True

        if st.session_state.recording_started:
            duration = st.number_input("Recording duration (seconds)", min_value=1, max_value=10, value=5)
            if st.button("Start Recording"):
                audio_data, sample_rate = record_audio(duration)
                audio_file_path = "temp_audio.wav"
                save_wav(audio_data, sample_rate, audio_file_path)
                st.session_state.question = speech_to_text(audio_file_path)
                st.write("Recorded Question: ", st.session_state.question)
                st.session_state.recording_started = False

        if st.session_state.question:
            if st.button("Get Answer"):
                raw_answer = generate_answer(st.session_state.cleaned_text, st.session_state.question)
                refined_answer = refine_answer(raw_answer, st.session_state.question)
                st.session_state.answer = refined_answer
                st.write("## Answer:")
                st.write(f"**Question:** {st.session_state.question}")
                st.write(f"**Answer:** {refined_answer}")
                st.session_state.audio_file = text_to_speech(refined_answer)

            if st.session_state.answer:
                st.write("## Answer:")
                st.write(f"**Question:** {st.session_state.question}")
                st.write(f"**Answer:** {st.session_state.answer}")
                if st.button("Listen to the answer"):
                    audio_bytes = open(st.session_state.audio_file, "rb").read()
                    st.audio(audio_bytes, format="audio/mp3")

                st.write("## Translate Answer:")
                tgt_lang_answer = st.selectbox("Select target language", ["es", "fr", "de", "it", "pt", "zh"], key="answer_translate")
                if st.button("Translate Answer"):
                    translated_answer = translate_text(st.session_state.answer, "en", tgt_lang_answer)
                    st.session_state.translated_answer = translated_answer
                    st.write("## Translated Answer:")
                    st.write(translated_answer)
                    if st.button("Listen to Translated Answer"):
                        audio_file_translated_answer = text_to_speech(translated_answer)
                        audio_bytes_translated_answer = open(audio_file_translated_answer, "rb").read()
                        st.audio(audio_bytes_translated_answer, format="audio/mp3")

if __name__ == "__main__":
    main()