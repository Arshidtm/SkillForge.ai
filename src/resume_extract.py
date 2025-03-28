import streamlit as st
import spacy
from pathlib import Path
import os 
from PyPDF2 import PdfReader


# Navigate up from src/ to root, then to model
# model_path = "..\\resume_ner_model"



# nlp = spacy.load(model_path)




def extract_resume(file): 
    pdf_reader = PdfReader(file)
    full_text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:  # Only add if text exists
            full_text += page_text + "\n"
    return full_text


def extract_details(text,nlp):
    processed_text = nlp(text)
    
    extract_skills = " ".join([ent.text for ent in processed_text.ents if ent.label_ == 'SKILLS' or ent.label_ == "CERTIFICATION" or ent.label_ == "DEGREE"])
    return extract_skills

