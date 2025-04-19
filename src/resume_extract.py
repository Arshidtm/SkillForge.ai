import logging
import streamlit as st 
from typing import Optional,List
from PyPDF2 import PdfReader
import json
import re
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate


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
            return "Please provide a valid PDF file"

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


def text_from_jd(docs: List[Document]) -> str:
    """
    Extracts the first half of job description text from a list of Document objects.

    The function concatenates the `page_content` from the first half of the documents
    and returns it as a single text string.

    Args:
        docs (List[Document]): A list of Document objects containing job descriptions.

    Returns:
        str: Concatenated text from the first half of the documents.

    Raises:
        ValueError: If the input is not a list or is empty.
        AttributeError: If an item in the list does not contain 'page_content'.
    """
    try:
        if not docs or not isinstance(docs, list):
            raise ValueError("Input must be a non-empty list of Document objects.")

        text = ""
        mid = len(docs) // 2

        for i, doc in enumerate(docs):
            if not hasattr(doc, "page_content"):
                raise AttributeError(f"Document at index {i} is missing 'page_content'.")
            text += doc.page_content
            if i == mid:
                break

        logger.info(f"Extracted text from Documents")
        return text

    except Exception as e:
        logger.error(f"Failed to extract text from job descriptions: {e}")
        raise


def parsed_skills(text: str, prompt_template: str, llm) -> List[str]:
    """
    Extracts skills from  text using an LLM prompt and returns a combined list of programming
    languages and tools/frameworks.

    Args:
        text (str): Raw resume text.
        prompt_template (str): Prompt template with a placeholder for text context.
        llm: Language model object with `.invoke()` method (e.g., from LangChain).

    Returns:
        List[str]: List of extracted skill names.

    Raises:
        ValueError: If the model response cannot be parsed as JSON or expected fields are missing.
    """
    try:
        # Prepare and send prompt
        logger.info("Formatting and sending prompt to LLM...")
        prompt = ChatPromptTemplate.from_template(prompt_template)
        formatted_prompt = prompt.format(text=text)

        response = llm.invoke(formatted_prompt)
        parsed_data = response.content

        # Clean and load JSON from LLM response
        logger.info("Cleaning and parsing LLM response...")
        cleaned = re.sub(r"^```json|```$", "", parsed_data.strip(), flags=re.MULTILINE)
        dict_parsed_data = json.loads(cleaned)

        # Extract skills
        logger.info(f"Extracting skills from parsed JSON... {dict_parsed_data}")
        skills = dict_parsed_data.get('skills', {})       

        logger.info(f"✅ Successfully extracted {len(skills)} skills.")
        return skills

    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from LLM response: {e}")
        raise ValueError("Invalid JSON format returned by LLM.")
    except KeyError as e:
        logger.error(f"Expected key missing in parsed data: {e}")
        raise ValueError(f"Missing expected key in LLM output: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error while extracting skills: {e}")
        raise
