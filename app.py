from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for
from transformers import T5ForConditionalGeneration, T5Tokenizer, MarianMTModel, MarianTokenizer
import spacy
import fitz  # PyMuPDF
from PIL import Image
from gtts import gTTS
import tempfile
import speech_recognition as sr
import sounddevice as sd
import scipy.io.wavfile as wav
import ocr_module
import re
from send_email import send_email

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/audio'

# Load T5 model and tokenizer
model_name = "t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open("pdf", pdf_file.read()) as doc:
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
    return text

def clean_text(text):
    cleaned_text = re.sub(r'-\s+', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = cleaned_text.strip()
    return cleaned_text

def summarize_text(text):
    input_text = f"summarize: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs.input_ids, max_length=150, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

def generate_answer(text, question):
    input_text = f"question: {question} context: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs.input_ids, max_length=150, num_beams=4, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def refine_answer(answer, question):
    doc = nlp(answer)
    key_phrases = [chunk.text for chunk in doc.noun_chunks if chunk.text.lower() in question.lower()]
    entities = [ent.text for ent in doc.ents if ent.text.lower() in question.lower()]
    refined_answer = ' '.join(set(key_phrases + entities))
    return refined_answer if refined_answer else answer

def text_to_speech(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", dir=app.config['UPLOAD_FOLDER']) as temp_audio_file:
        tts.save(temp_audio_file.name)
        return temp_audio_file.name

def translate_text(text, src_lang, tgt_lang):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    translated_ids = model.generate(inputs["input_ids"], max_length=512, num_beams=4, early_stopping=True)
    translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    return translated_text

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    pdf_file = request.files['pdf']
    raw_text = extract_text_from_pdf(pdf_file)
    cleaned_text = clean_text(raw_text)
    return jsonify({'text': cleaned_text})

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form['text']
    summary = summarize_text(text)
    return jsonify({'summary': summary})

@app.route('/translate', methods=['POST'])
def translate():
    text = request.form['text']
    src_lang = request.form['src_lang']
    tgt_lang = request.form['tgt_lang']
    translated_text = translate_text(text, src_lang, tgt_lang)
    return jsonify({'translated_text': translated_text})

@app.route('/ask_question', methods=['POST'])
def ask_question():
    text = request.form['text']
    question = request.form['question']
    raw_answer = generate_answer(text, question)
    refined_answer = refine_answer(raw_answer, question)
    return jsonify({'answer': refined_answer})

@app.route('/text_to_speech', methods=['POST'])
def convert_text_to_speech():
    text = request.form['text']
    audio_file_path = text_to_speech(text)
    return jsonify({'audio_file': audio_file_path})

@app.route('/speech_to_text', methods=['POST'])
def convert_speech_to_text():
    audio_file = request.files['audio']
    text = speech_to_text(audio_file)
    return jsonify({'text': text})

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        user_email = request.form['email']
        raw_message = request.form['message']
        message = f"Subject: New email from {user_email}\n\nFrom: {user_email}\n{raw_message}"
        send_email(message)
        return render_template('contact.html', success=True)
    return render_template('contact.html')

if __name__ == "__main__":
    app.run(debug=True)