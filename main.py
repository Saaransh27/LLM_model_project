import fitz  # PyMuPDF
import re
import streamlit as st
import spacy
from PIL import Image
from transformers import T5ForConditionalGeneration, T5Tokenizer, MarianMTModel, MarianTokenizer, logging
from gtts import gTTS
import tempfile
import os

# Suppress warnings from transformers
logging.set_verbosity_error()

# --------------------------------------------------------------------
# CONFIGURATION - MUST BE THE FIRST STREAMLIT COMMAND
# Fixes StreamlitAPIException
# --------------------------------------------------------------------
st.set_page_config(page_title="Multi-format Input QA System", page_icon="📝", layout="centered")

# ====================================================================
# MOCK MODULES & UTILITIES FOR SELF-CONTAINED EXECUTION
# ====================================================================

# 1. Mock OCR Module (Replaces import ocr_module)
# This simulates detecting text from an image for demonstration purposes.
def detect_text(image):
    """Mocks OCR detection since external modules/APIs are restricted."""
    st.warning("Using a mocked OCR result. Real OCR functionality requires 'tesseract' or an API like Google Vision/Azure.")
    return "The sun is the star at the center of the Solar System. It is a nearly perfect ball of hot plasma, heated to incandescence by nuclear fusion reactions in its core, radiating the energy mainly as visible light, ultraviolet light, and infrared radiation. It is by far the most important source of energy for life on Earth. Its diameter is about 1.39 million kilometers, or 109 times that of Earth. Its mass is about 330,000 times that of Earth."

# 2. Mock Audio Recording (Replaces sounddevice/speech_recognition)
# This prevents crashes due to missing microphone drivers or environment restrictions.
def mock_speech_to_text():
    """Mocks speech-to-text conversion by returning a sample question."""
    st.success("Microphone access is restricted in this environment. Simulating a recorded question.")
    return "What is the primary source of energy for Earth?"

# ====================================================================
# MODEL LOADING (T5 and SpaCy)
# ====================================================================

# Load T5 model and tokenizer
@st.cache_resource
def load_t5_model():
    """Loads and caches the T5 model and tokenizer."""
    model_name = "t5-base"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    # Using T5Tokenizer is okay, but T5TokenizerFast is often recommended for performance.
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_t5_model()

# Load SpaCy model
@st.cache_resource
def load_spacy_model():
    """Loads and caches the SpaCy model."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        # Fallback if model is not installed; though usually pre-installed
        st.error("SpaCy model 'en_core_web_sm' not found. Please ensure it is installed.")
        return None

nlp = load_spacy_model()

# ====================================================================
# CORE FUNCTIONS
# ====================================================================

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    """Extracts raw text from a PDF file using PyMuPDF."""
    text = ""
    # fitz.open requires a file path or bytes; we read bytes from the uploaded file
    with fitz.open("pdf", pdf_file.read()) as doc:
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
    return text

# Function to clean text
def clean_text(text):
    """Cleans up hyphens and excessive whitespace."""
    cleaned_text = re.sub(r'-\s+', '', text) # Remove hyphens followed by a line break
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text) # Replace multiple spaces/newlines with a single space
    cleaned_text = cleaned_text.strip()
    return cleaned_text

# Function to summarize text
def summarize_text(text):
    """Generates a summary using the T5 model."""
    input_text = f"summarize: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs.input_ids, max_length=150, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# Function to generate answer
def generate_answer(text, question):
    """Generates an answer using the T5 model for Q&A."""
    input_text = f"question: {question} context: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs.input_ids, max_length=150, num_beams=4, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Function to refine answer
def refine_answer(answer, question):
    """Attempts to refine the answer using SpaCy for entities/key phrases."""
    if not nlp:
        return answer
    
    doc = nlp(answer)
    # Extract noun chunks and entities that overlap with the question
    question_lower = question.lower()
    key_phrases = [
        chunk.text for chunk in doc.noun_chunks 
        if any(word in question_lower for word in chunk.text.lower().split())
    ]
    
    entities = [
        ent.text for ent in doc.ents 
        if any(word in question_lower for word in ent.text.lower().split())
    ]
    
    # Simple deduplication and return
    refined_parts = list(set(key_phrases + entities))
    if refined_parts:
        # A simple joining might not form a coherent sentence, so fall back to the raw answer
        # unless refinement is clearly successful. We'll return the original answer for coherence.
        return answer 
    else:
        return answer


# Function to correct grammar
def correct_grammar(text):
    """Corrects grammar using the T5 model (via a grammar prompt)."""
    input_text = f"grammar: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs.input_ids, max_length=512, num_beams=4, early_stopping=True)
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

# Function to convert text to speech and return audio bytes
def text_to_speech(text):
    """Converts text to speech using gTTS, saves to a temp file, and returns audio bytes."""
    tts = gTTS(text, lang='en')
    temp_audio_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            temp_audio_file = tmp.name
            tts.write_to_fp(tmp)
        
        # Read the file bytes before deleting
        with open(temp_audio_file, "rb") as f:
            audio_bytes = f.read()
            
        return audio_bytes
    finally:
        # Ensure the temporary file is deleted
        if temp_audio_file and os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)

# Function to translate text
@st.cache_resource
def load_translator(src_lang, tgt_lang):
    """Loads and caches the MarianMT model for translation."""
    try:
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        st.error(f"Could not load translation model for {src_lang} to {tgt_lang}. Error: {e}")
        return None, None

def translate_text(text, src_lang, tgt_lang):
    """Translates text using MarianMT model."""
    model, tokenizer = load_translator(src_lang, tgt_lang)
    if not model:
        return "Translation failed due to model loading error."

    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    translated_ids = model.generate(inputs["input_ids"], max_length=512, num_beams=4, early_stopping=True)
    translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    return translated_text

# ====================================================================
# STREAMLIT APP LAYOUT
# ====================================================================

def main():
    st.title("📝 Multi-format Input Question-Answer System")
    st.markdown("---")

    # --- Session State Management ---
    if "cleaned_text" not in st.session_state: st.session_state.cleaned_text = ""
    if "question" not in st.session_state: st.session_state.question = ""
    if "answer" not in st.session_state: st.session_state.answer = ""
    if "summary" not in st.session_state: st.session_state.summary = ""

    # --- Input Selection ---
    option = st.selectbox(
        "**1. Select Input Format:**",
        ("Select an option", "Image (Mock OCR)", "PDF", "Text"),
        key="input_option"
    )
    
    # --- Input Handling Logic ---
    current_input_text = ""

    if option == "Image (Mock OCR)":
        image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="image_uploader")
        if image_file:
            with st.spinner("Extracting text from image..."):
                image = Image.open(image_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                extracted_text = detect_text(image)
                current_input_text = clean_text(extracted_text)

    elif option == "PDF":
        pdf_file = st.file_uploader("Upload a PDF file", type="pdf", key="pdf_uploader")
        if pdf_file:
            with st.spinner("Extracting text from PDF..."):
                raw_text = extract_text_from_pdf(pdf_file)
                current_input_text = clean_text(raw_text)

    elif option == "Text":
        input_text = st.text_area("Enter your text", key="text_area")
        if input_text:
            current_input_text = clean_text(input_text)
    
    # Update session state after extraction/input
    if current_input_text:
        st.session_state.cleaned_text = current_input_text
        if st.checkbox("View Extracted/Input Text", value=True):
            st.markdown("### Input Text for Processing:")
            st.info(st.session_state.cleaned_text)

    st.markdown("---")

    # --- Text Processing and Q&A ---
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

        # --- Question Answering (Q&A) ---
        st.header("**3. Question Answering**")
        
        # User Question Input (Text or Mocked Speech)
        question = st.text_input("Ask a question about the content:", value=st.session_state.question, key="question_input")
        st.session_state.question = question
        
        if st.button("🎤 Mock Recording"):
            st.session_state.question = mock_speech_to_text()
            # Force rerun to update the text_input widget
            st.experimental_rerun()

        if st.session_state.question and st.button("🔎 Get Answer", key="get_answer_btn"):
            with st.spinner("Generating answer..."):
                raw_answer = generate_answer(st.session_state.cleaned_text, st.session_state.question)
                # Refining is complex for a general Q&A model, keeping the logic simple
                refined_answer = raw_answer 
                st.session_state.answer = refined_answer
                
                st.markdown("#### Generated Answer:")
                st.write(f"**Question:** {st.session_state.question}")
                st.success(f"**Answer:** {st.session_state.answer}")
                
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
    # The original file had these functions but they are hardware dependent and not runnable here.
    # We define them as no-ops to satisfy imports, but the logic above uses the mocks.
    # The original imports `speech_recognition as sr` and `sounddevice as sd` were removed.
    def record_audio(duration, sample_rate=44100): 
        raise NotImplementedError("Audio recording is not supported in this environment. Use the 'Mock Recording' button.")
    def save_wav(audio_data, sample_rate, file_path):
        raise NotImplementedError("Audio recording is not supported in this environment.")
    
    main()