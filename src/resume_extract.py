import logging
import streamlit as st 
from typing import Optional
from PyPDF2 import PdfReader


# Navigate up from src/ to root, then to model
# model_path = "..\\resume_ner_model"
# nlp = spacy.load(model_path)

# configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@st.cache_data(ttl=600) 
def extract_resume(file_path: str) -> Optional[str]:
    """
    Extract text from a PDF resume file.

    Args:
        file_path (str): Path to the resume file

    Returns:
        Optional[str]: Extracted text if successful, otherwise None
    """
    try:

        pdf_reader = PdfReader(file_path)
        full_text = ""

        for page in pdf_reader.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
            except Exception as e:
                logger.warning(f"Error extracting text from pages: {str(e)}")
                continue

        if not full_text.strip():
            logger.warning("No text was extracted from the resume")
            return None

        return full_text
    except FileNotFoundError:
        logger.error(f"File not found: {str(file_path)}")
    except Exception as e:
        logger.error(f"Unexpected Error processsing PDF: {str(e)}")
    return None


def extract_details(text: str, nlp) -> Optional[str]:
    """
    Extract skills, certifications, and degree from the resume using custom trained spacy model

    Args:
        text (str): The resume text to process
        nlp (_type_): The loaded spaCy NLP model

    Returns:
        Optional[str]: Extracted text if successful, None otherwise.
    """

    if not text or not isinstance(text, str):
        logger.error("Invalid input test for model to process")
        return None

    try:
        processed_text = nlp(text)

        extract_skills = "This is my extracted data from my resume \n" + " ".join(
            [
                ent.text
                for ent in processed_text.ents
                if ent.label_ == "SKILLS"
                or ent.label_ == "CERTIFICATION"
                or ent.label_ == "DEGREE"
            ]
        )
        if not extract_skills:
            logger.info("No skills/certification/degree found in the text")

        return extract_skills if extract_skills else None
    except Exception as e:
        logger.error(f"Error during NLP processing: {e}")
        return None
