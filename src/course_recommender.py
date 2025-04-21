import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd 
import logging
import pickle

# configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_dataset(file_path):
    """Loads the combined_courses.csv file and returns a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        logging.info("Dataset loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return None
    

def load_skill_embeddings(file_path):
    """Loads precomputed skill embeddings from a pickle file."""
    try:
        with open(file_path, "rb") as f:
            skill_embeddings = pickle.load(f)
            logging.info("Skill embeddings loaded successfully.")
            return skill_embeddings
    except Exception as e:
        logging.error(f"Error loading skill embeddings: {e}")
        return None


def recommend_courses(vector_store, nlp, resume_skills, embeddings, skill_embeddings, df, top_n=3):
    """
    Optimized course recommendation based on skill gaps with reduced time complexity.
    
    Args:
        vector_store: FAISS vector store with job data
        nlp: SpaCy NLP model for skill extraction
        resume_skills: List of skills from resume (lowercase)
        embeddings: Embedding model (same as used in vector store) HuggingFaceEmbeddings
        skill_embeddings: Precomputed embeddings for skills
        df: DataFrame containing course information
        top_n: Number of courses to recommend
    
    Returns:        
        - recommended_courses: List of recommended courses       
    """
    try:
        # Convert resume skills to set for O(1) lookups
        if resume_skills is not None:
            resume_skills_set = set(skill.lower() for skill in resume_skills)        

        # Extract skills from vector store
        skill_gaps = set()
        for doc in vector_store.docstore._dict.values():
            doc_skills = [
                ent.text.lower() for ent in nlp(doc.page_content).ents 
                if ent.label_ in ('SKILLS', 'CERTIFICATION')
            ]
            skill_gaps.update(skill for skill in doc_skills if skill not in resume_skills_set)
        
        logger.info(f"Identified skill gaps: {skill_gaps}")
        
        if not skill_gaps:
            logger.warning("No skill gaps found. Returning empty result.")
            return "No skill gap found, Recommend some course to enhance skill"

        # Embed the skill gap text
        skill_gap_text = " ".join(skill_gaps)
        skill_gap_embedding = embeddings.embed_query(skill_gap_text)
        logger.info("Skill gap embedding generated.")

        # Compute cosine similarity
        similarity_scores = cosine_similarity(
            [skill_gap_embedding],
            skill_embeddings
        )[0]

        # Get top N course indices
        top_indices = np.argpartition(similarity_scores, -top_n)[-top_n:]
        top_indices = top_indices[np.argsort(similarity_scores[top_indices])[::-1]]

        # Prepare course recommendation results
        recommended_courses = []
        for idx in top_indices:
            course = {
                "title": df.iloc[idx].get('Title', 'N/A'),
                "organization": df.iloc[idx].get('Organization', 'N/A'),
                "platform": df.iloc[idx].get('Platform', 'N/A')                
            }
            recommended_courses.append(course)

        logger.info("Top recommended courses generated.")

        return recommended_courses            

    except Exception as e:
        logger.error(f"Error in recommend_courses: {str(e)}", exc_info=True)
        raise


def recommend_courses_llm(jd_skills,resume_skills, embeddings, skill_embeddings, df, top_n=3):
    """
    Optimized course recommendation based on skill gaps with reduced time complexity.
    
    Args:
           
        jd_skills: List of skills from job_data         
        resume_skills: List of skills from resume (lowercase)
        embeddings: Embedding model (same as used in vector store) HuggingFaceEmbeddings
        skill_embeddings: Precomputed embeddings for skills
        df: DataFrame containing course information
        top_n: Number of courses to recommend
    
    Returns:        
        - recommended_courses: List of recommended courses       
    """
    try:
        # Convert resume skills to set for O(1) lookups
        resume_skills_set = set()
        if resume_skills:
            resume_skills_set = set(skill.lower() for skill in resume_skills)        

        # Extract skills from vector store
        jd_skills_set = set(skill.lower() for skill in jd_skills)
        
        skill_gaps = jd_skills_set - resume_skills_set      
                
        logger.info(f"Identified skill gaps: {skill_gaps}")
        
        if not skill_gaps:
            logger.warning("No skill gaps found. Returning empty result.")
            return "No skill gap found, Recommend some course to enhance skill"

        # Embed the skill gap text
        skill_gap_text = " ".join(skill_gaps)
        skill_gap_embedding = embeddings.embed_query(skill_gap_text)
        skill_embeddings = np.array(skill_embeddings).astype(np.float32)
        skill_gap_embedding = np.array(skill_gap_embedding, dtype=np.float32)


        logger.info("Skill gap embedding generated.")

        # Compute cosine similarity
        similarity_scores = cosine_similarity(
            [skill_gap_embedding],
            skill_embeddings
        )[0]

        # Get top N course indices
        top_indices = np.argpartition(similarity_scores, -top_n)[-top_n:]
        top_indices = top_indices[np.argsort(similarity_scores[top_indices])[::-1]]

        # Prepare course recommendation results
        recommended_courses = []
        for idx in top_indices:
            course = {
                "title": df.iloc[idx].get('Title', 'N/A'),
                "organization": df.iloc[idx].get('Organization', 'N/A'),
                "platform": df.iloc[idx].get('Platform', 'N/A')                
            }
            recommended_courses.append(course)

        logger.info(f"Top recommended courses generated. {recommended_courses}")

        return recommended_courses            

    except Exception as e:
        logger.error(f"Error in recommend_courses: {str(e)}", exc_info=True)
        raise