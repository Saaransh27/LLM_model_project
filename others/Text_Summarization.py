# Step 1. Importing Libraries

import sys
import math
import bs4 as bs
import urllib.request
import re
import PyPDF2
import nltk
from nltk.stem import WordNetLemmatizer
import spacy

# Execute this line if you are running this code for the first time
nltk.download('wordnet')

# Initializing few variables
nlp = spacy.load('en_core_web_sm')
lemmatizer = WordNetLemmatizer()

# Step 2. Define functions for Reading Input Text

# Function to Read .txt File and return its Text
def file_text(filepath):
    with open(filepath) as f:
        text = f.read().replace("\n", '')
        return text

# Function to Read PDF File and return its Text
def pdf_reader(pdf_path):
    with open(pdf_path, 'rb') as pdfFileObject:
        pdf_reader = PyPDF2.PdfReader(pdfFileObject)
        count = len(pdf_reader.pages)
        print("\nTotal Pages in pdf = ", count)

        c = input("Do you want to read the entire pdf? [Y]/N  :  ")
        if c.lower() == 'n':
            start_page = int(input("Enter start page number (Indexing start from 0) :  "))
            end_page = int(input(f"Enter end page number (Less than {count}) : "))
            
            if start_page < 0 or start_page >= count:
                print("\nInvalid Start page given")
                sys.exit()
                
            if end_page < 0 or end_page >= count:
                print("\nInvalid End page given")
                sys.exit()
                
        else:
            start_page, end_page = 0, count - 1

        text = ""
        for i in range(start_page, end_page + 1):
            page = pdf_reader.pages[i]
            text += page.extract_text()
        
        return text

# Function to Read Wikipedia page URL and return its Text   

input_text_type = int(input("Select one way of inputting your text: \n1. Type your Text (or Copy-Paste)\n2. Load from .txt file\n3. Load from .pdf file\n\n"))

if input_text_type == 1:
    text = input("Enter your text: \n\n")

elif input_text_type == 2:
    txt_path = input("Enter file path: ")
    text = file_text(txt_path)
    
elif input_text_type == 3:
    file_path = input("Enter file path: ")
    text = pdf_reader(file_path)
    
    
else:
    print("Sorry! Wrong Input, Try Again.")
    sys.exit()

# Step 4. Defining functions to create Tf-Idf Matrix

# Function to calculate frequency of word in each sentence
def frequency_matrix(sentences):
    freq_matrix = {}
    stop_words = nlp.Defaults.stop_words

    for sent in sentences:
        freq_table = {}
        words = [word.text.lower() for word in sent if word.text.isalnum()]
       
        for word in words:
            word = lemmatizer.lemmatize(word)
            if word not in stop_words:
                if word in freq_table:
                    freq_table[word] += 1
                else:
                    freq_table[word] = 1

        freq_matrix[sent[:15]] = freq_table

    return freq_matrix

# Function to calculate Term Frequency (TF) of each word
def tf_matrix(freq_matrix):
    tf_matrix = {}

    for sent, freq_table in freq_matrix.items():
        tf_table = {}
        total_words_in_sentence = len(freq_table)
        for word, count in freq_table.items():
            tf_table[word] = count / total_words_in_sentence

        tf_matrix[sent] = tf_table

    return tf_matrix

# Function to find how many sentences contain a 'word'
def sentences_per_words(freq_matrix):
    sent_per_words = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in sent_per_words:
                sent_per_words[word] += 1
            else:
                sent_per_words[word] = 1

    return sent_per_words

# Function to calculate Inverse Document frequency (IDF) for each word
def idf_matrix(freq_matrix, sent_per_words, total_sentences):
    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_sentences / float(sent_per_words[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix

# Function to calculate Tf-Idf score of each word
def tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):
        tf_idf_table = {}
        for (word1, tf_value), (word2, idf_value) in zip(f_table1.items(), f_table2.items()):
            tf_idf_table[word1] = float(tf_value * idf_value)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix

# Function to rate every sentence with some score calculated on basis of Tf-Idf
def score_sentences(tf_idf_matrix):
    sentence_score = {}

    for sent, f_table in tf_idf_matrix.items():
        total_tfidf_score_per_sentence = sum(f_table.values())
        total_words_in_sentence = len(f_table)
        if total_words_in_sentence != 0:
            sentence_score[sent] = total_tfidf_score_per_sentence / total_words_in_sentence

    return sentence_score

# Function Calculating average sentence score 
def average_score(sentence_score):
    total_score = sum(sentence_score.values())
    average_sent_score = (total_score / len(sentence_score))
    return average_sent_score

# Function to return summary of article
def create_summary(sentences, sentence_score, threshold):
    summary = ''
    for sentence in sentences:
        if sentence[:15] in sentence_score and sentence_score[sentence[:15]] >= (threshold):
            summary += " " + sentence.text
    return summary

# Step 5. Using all functions to generate summary

# Counting number of words in original article
original_words = text.split()
original_words = [w for w in original_words if w.isalnum()]
num_words_in_original_text = len(original_words)

# Converting received text into spaCy Doc object
doc = nlp(text)

# Extracting all sentences from the text in a list
sentences = list(doc.sents)
total_sentences = len(sentences)

# Generating Frequency Matrix
freq_matrix = frequency_matrix(sentences)

# Generating Term Frequency Matrix
tf_matrix = tf_matrix(freq_matrix)

# Getting number of sentences containing a particular word
num_sent_per_words = sentences_per_words(freq_matrix)

# Generating ID Frequency Matrix
idf_matrix = idf_matrix(freq_matrix, num_sent_per_words, total_sentences)

# Generating Tf-Idf Matrix
tf_idf_matrix = tf_idf_matrix(tf_matrix, idf_matrix)

# Generating Sentence score for each sentence
sentence_scores = score_sentences(tf_idf_matrix)

# Setting threshold to average value (You are free to play with the values)
threshold = average_score(sentence_scores)

# Getting summary 
summary = create_summary(sentences, sentence_scores, 1.3 * threshold)

print("\n\n")
print("*" * 20, "Summary", "*" * 20)
print("\n")
print(summary)
print("\n\n")
print("Total words in original article = ", num_words_in_original_text)
print("Total words in summarized article = ", len(summary.split()))