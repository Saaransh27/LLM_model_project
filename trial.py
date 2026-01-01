import fitz  # PyMuPDF
import re
import streamlit as st
from PIL import Image
from transformers import T5ForConditionalGeneration, T5Tokenizer, MarianMTModel, MarianTokenizer, logging
from gtts import gTTS
import tempfile
import os

logging.set_verbosity_error()

st.set_page_config(page_title="Multi-format Input QA System", page_icon="📝", layout="centered")

def detect_text(image):
    """Mocks OCR detection since external modules/APIs are restricted."""
    st.warning("Using a mocked OCR result. Real OCR functionality requires 'tesseract' or an API like Google Vision/Azure.")
    return "The sun is the star at the center of the Solar System. It is a nearly perfect ball of hot plasma, heated to incandescence by nuclear fusion reactions in its core, radiating the energy mainly as visible light, ultraviolet light, and infrared radiation. It is by far the most important source of energy for life on Earth. Its diameter is about 1.39 million kilometers, or 109 times that of Earth. Its mass is about 330,000 times that of Earth."

def mock_speech_to_text():
    st.success("Microphone access is restricted. Simulating a recorded question.")
    return "What is the main concept discussed in the document?"


@st.cache_resource
def load_t5_model():
    """Loads and caches the T5 model and tokenizer."""
    model_name = "t5-base"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_t5_model()

@st.cache_resource
def load_translator(src_lang, tgt_lang):
    try:
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        st.error(f"Could not load translation model for {src_lang} to {tgt_lang}. Error: {e}")
        return None, None


def extract_text_from_pdf(pdf_file):
    """Extracts raw text from a PDF file using PyMuPDF."""
    text = ""
    with fitz.open("pdf", pdf_file.read()) as doc:
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
    return text

def clean_text(text):
    cleaned_text = re.sub(r'-\s+', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text.strip()

def text_to_speech(text):
    tts = gTTS(text, lang='en')
    temp_audio_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            temp_audio_file = tmp.name
            tts.write_to_fp(tmp)
        
        with open(temp_audio_file, "rb") as f:
            audio_bytes = f.read()
            
        return audio_bytes
    finally:
        if temp_audio_file and os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)

def summarize_text(text):
    input_text = f"summarize: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs.input_ids, max_length=150, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

def generate_answer(text, question):
    # Strong prompt instruction to force a detailed explanation
    input_text = f"Explain the concept of {question} in detail based on the following context: {text}"
    
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    outputs = model.generate(
        inputs.input_ids, 
        max_length=300,             # Max output tokens
        num_beams=8,                # High beam search for quality
        length_penalty=2.0,         # Strongly encourages longer responses
        early_stopping=True
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def correct_grammar(text):
    input_text = f"grammar: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs.input_ids, max_length=512, num_beams=4, early_stopping=True)
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

def translate_text(text, src_lang, tgt_lang):
    model_mt, tokenizer_mt = load_translator(src_lang, tgt_lang)
    if not model_mt:
        return "Translation failed due to model loading error."

    inputs = tokenizer_mt(text, return_tensors="pt", max_length=512, truncation=True)
    translated_ids = model_mt.generate(inputs["input_ids"], max_length=512, num_beams=4, early_stopping=True)
    translated_text = tokenizer_mt.decode(translated_ids[0], skip_special_tokens=True)
    return translated_text

def main():
    st.title("📝 Multi-format Input Question-Answer System")
    st.markdown("---")

    if "cleaned_text" not in st.session_state: st.session_state.cleaned_text = ""
    if "question" not in st.session_state: st.session_state.question = ""
    if "answer" not in st.session_state: st.session_state.answer = ""
    if "summary" not in st.session_state: st.session_state.summary = ""

    # --- 1. Document Input ---
    st.header("**1. Document Input**")
    option = st.selectbox(
        "Select your input method:",
        ("PDF", "Text", "Image (Mock OCR)"),
        key="input_option",
        index=0 # Default to PDF
    )
    
    current_input_text = ""

    if option == "Image (Mock OCR)":
        image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="image_uploader")
        if image_file:
            with st.spinner("Extracting text from image..."):
                image = Image.open(image_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                current_input_text = clean_text(detect_text(image))

    elif option == "PDF":
        pdf_file = st.file_uploader("Upload a PDF file", type="pdf", key="pdf_uploader")
        if pdf_file:
            with st.spinner("Extracting text from PDF..."):
                current_input_text = clean_text(extract_text_from_pdf(pdf_file))

    elif option == "Text":
        input_text = st.text_area("Enter your text", key="text_area")
        if input_text:
            current_input_text = clean_text(input_text)
    
    # Update state and display extracted text
    if current_input_text:
        st.session_state.cleaned_text = current_input_text
        with st.expander("View Extracted/Input Text"):
            st.markdown("### Input Text for Processing:")
            st.info(st.session_state.cleaned_text)

    st.markdown("---")

    # --- 2. Text Processing Tools ---
    if st.session_state.cleaned_text:
        
        st.header("**2. Processing Tools**")
        
        # --- Grammar Correction ---
        if st.checkbox("Correct Grammar", key="correct_grammar_check"):
            with st.spinner("Correcting grammar..."):
                corrected_text = correct_grammar(st.session_state.cleaned_text)
                st.markdown("#### Corrected Text:")
                st.success(corrected_text)
                st.session_state.cleaned_text = corrected_text # Use corrected text for next steps

        # --- Summarization ---
        st.subheader("Summarization and Translation")
        if st.checkbox("Summarize the content", key="summarize_check"):
            with st.spinner("Generating summary..."):
                st.session_state.summary = summarize_text(st.session_state.cleaned_text)
                st.markdown("#### Summary:")
                st.write(st.session_state.summary)

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🔊 Listen to Summary"):
                    with st.spinner("Generating audio..."):
                        audio_bytes = text_to_speech(st.session_state.summary)
                        st.audio(audio_bytes, format="audio/mp3")

            # --- Summary Translation ---
            with col2:
                tgt_lang = st.selectbox("Translate Summary to:", ["es", "fr", "de", "it", "pt", "zh"], key="summary_lang")
                if st.button(f"🌐 Translate Summary to {tgt_lang.upper()}"):
                    with st.spinner(f"Translating summary to {tgt_lang.upper()}..."):
                        translated_summary = translate_text(st.session_state.summary, "en", tgt_lang)
                        st.markdown(f"#### Translated Summary ({tgt_lang.upper()}):")
                        st.info(translated_summary)
                        if st.button(f"🔊 Listen to {tgt_lang.upper()} Summary", key="listen_summary_translated"):
                            with st.spinner("Generating audio..."):
                                audio_bytes_translated = text_to_speech(translated_summary)
                                st.audio(audio_bytes_translated, format="audio/mp3")
        
        st.markdown("---")

        # --- 3. Question Answering (Q&A) ---
        st.header("**3. Question Answering**")
        
        st.markdown("##### **Method 1: Type your question**")
        question = st.text_input("Enter your question:", value=st.session_state.question, key="question_input", label_visibility="collapsed")
        st.session_state.question = question
        
        st.markdown("##### **Method 2: Use Mock Audio Input**")
        if st.button("🎤 Mock Recording"):
            st.session_state.question = mock_speech_to_text()
            st.experimental_rerun()

        if st.button("🔎 Get Answer", key="get_answer_btn"):
            if st.session_state.question:
                with st.spinner("Generating answer..."):
                    st.session_state.answer = generate_answer(st.session_state.cleaned_text, st.session_state.question)
                    
                    st.markdown("#### Generated Answer:")
                    st.write(f"**Question:** {st.session_state.question}")
                    st.success(f"**Answer:** {st.session_state.answer}")
            else:
                st.warning("Please enter a question or use the Mock Recording feature before clicking 'Get Answer'.")
                
        # --- Answer Output and Translation ---
        if st.session_state.answer:
            st.subheader("Answer Post-Processing")
            col3, col4 = st.columns([1, 1])

            with col3:
                if st.button("🔊 Listen to Answer", key="listen_answer"):
                    with st.spinner("Generating audio..."):
                        audio_bytes = text_to_speech(st.session_state.answer)
                        st.audio(audio_bytes, format="audio/mp3")
            
            with col4:
                tgt_lang_answer = st.selectbox("Translate Answer to:", ["es", "fr", "de", "it", "pt", "zh"], key="answer_lang")
                if st.button(f"🌐 Translate Answer to {tgt_lang_answer.upper()}"):
                    with st.spinner(f"Translating answer to {tgt_lang_answer.upper()}..."):
                        translated_answer = translate_text(st.session_state.answer, "en", tgt_lang_answer)
                        st.markdown(f"#### Translated Answer ({tgt_lang_answer.upper()}):")
                        st.info(translated_answer)
                        if st.button(f"🔊 Listen to {tgt_lang_answer.upper()} Answer", key="listen_answer_translated"):
                            with st.spinner("Generating audio..."):
                                audio_bytes_translated = text_to_speech(translated_answer)
                                st.audio(audio_bytes_translated, format="audio/mp3")


if __name__ == "__main__":
    # Define no-op functions to satisfy Streamlit if it were trying to use them
    def record_audio(duration, sample_rate=44100): 
        raise NotImplementedError("Audio recording is not supported in this environment.")
    def save_wav(audio_data, sample_rate, file_path):
        raise NotImplementedError("Audio recording is not supported in this environment.")
    
    main()