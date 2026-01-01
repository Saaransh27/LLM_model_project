# Palm use kra hh isme but abhi is code mei there is only user pdf input, text cleaning.
# Then, generation of Qna format using NLP techniques- NER and Dependency parsing for better generation of qna.
# iske bad training krenge which im searching itne isko apne system k hisab se modify krke try kr chlane ka.
# NEECHE STANFORD PARSER, MODEL KA PATH DAAL DIO CHLANE SE PHLE!!!

# kya hua??  just reading... okk

# jitna maine padha ye cheez toh har model se phle hogi toh ye toh kr hi lete hhn fir model training!? ok?

import fitz  # PyMuPDF
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.parse.stanford import StanfordDependencyParser

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Path to the Stanford NLP Parser (download and extract from https://stanfordnlp.github.io/CoreNLP/)
STANFORD_PARSER_PATH = r"C:\Users\LENEVO\Downloads\stanford-parser-4.2.0\stanford-parser.jar"
STANFORD_MODELS_PATH = r"C:\Users\LENEVO\Downloads\stanford-parser-4.2.0\stanford-parser-4.2.0-models.jar"

def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(pdf_file) as doc:
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
    return text

def clean_text(text):
    # Remove multiple spaces
    cleaned_text = re.sub(r'\s+', ' ', text)
    
    # Identify and remove references section
    references_start = cleaned_text.rfind("References")
    if references_start == -1:
        references_start = cleaned_text.rfind("Bibliography")
    
    if references_start != -1:
        cleaned_text = cleaned_text[:references_start]
    
    # Remove leading/trailing spaces
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text

def generate_qa_pairs_from_text(text):
    sentences = sent_tokenize(text)
    
    qa_pairs = []
    current_question = None
    current_answer = None
    
    for sentence in sentences:
        # Skip short or irrelevant sentences
        if len(sentence) < 10:
            continue
        
        # Attempt to identify questions based on patterns or heuristics
        if is_question(sentence):
            # If we have a previous question and answer, store them
            if current_question and current_answer:
                qa_pairs.append({"question": current_question, "answer": current_answer.strip()})
            
            # Start a new question
            current_question = sentence
            current_answer = ""
        else:
            # Append to the current answer
            if current_answer:
                current_answer += " " + sentence
            else:
                current_answer = sentence
    
    # Append the last question-answer pair
    if current_question and current_answer:
        qa_pairs.append({"question": current_question, "answer": current_answer.strip()})
    
    return qa_pairs

def is_question(sentence):
    # Simple heuristic to check if a sentence is a question
    if sentence.strip().endswith('?') or sentence.strip().startswith(('What', 'When', 'Where', 'Why', 'Who', 'How')):
        return True
    return False

def ner_tagging(sentence):
    # Perform NER tagging on a sentence
    words = word_tokenize(sentence)
    tagged_words = nltk.pos_tag(words)
    ne_tagged = ne_chunk(tagged_words)
    return ne_tagged

def dependency_parsing(sentence):
    # Perform dependency parsing on a sentence
    parser = StanfordDependencyParser(path_to_jar=STANFORD_PARSER_PATH, path_to_models_jar=STANFORD_MODELS_PATH)
    result = list(parser.raw_parse(sentence))
    parsed = result[0]  # Assuming only one parse per sentence
    return parsed

def main():
    # PDF input from the user
    pdf_file = input("Enter the path to your PDF file: ")
    
    raw_text = extract_text_from_pdf(pdf_file)
    
    cleaned_text = clean_text(raw_text)
    
    qa_pairs = generate_qa_pairs_from_text(cleaned_text)
    
    # Print generated Q&A pairs
    for idx, pair in enumerate(qa_pairs):
        print(f"Question {idx+1}: {pair['question']}")
        print(f"Answer {idx+1}: {pair['answer']}")
        print()
    
if __name__ == "__main__":
    main()