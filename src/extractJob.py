import http.client
import json
import random
import re
from langchain.schema import Document
from dotenv import load_dotenv
import os
import logging
import streamlit as st 

os.getcwd()
# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load the api keys
load_dotenv()

api_key = os.getenv("RAPIDAPI_KEY")

# Defining the constants
INDIAN_CITIES = [
    "Mumbai",
    "Delhi",
    "Bangalore",
    "Hyderabad",
    "Ahmedabad",
    "Chennai",
    "Kolkata",
    "Surat",
    "Pune",
    "Jaipur",
    "Lucknow",
    "Kanpur",
    "Nagpur",
    "Visakhapatnam",
    "Indore",
    "Thane",
    "Bhopal",
    "Patna",
    "Vadodara",
    "Ghaziabad",
]


@st.cache_data(ttl=600)
def fetch_jobs(query, location="India", results_wanted=5, api_key=api_key, strict_matching=True):
    """
    Fetch job descriptions from the API based on query and location with strict role matching.
    
    Args:
        query (str): Job role for searching
        location (str, optional): Location to search. Defaults to "India".
        results_wanted (int, optional): Number of job results desired. Defaults to 5.
        api_key (str, optional): Rapid API key for authentication. Defaults to api_key.
        strict_matching (bool, optional): Whether to enforce strict job title matching. Defaults to True.
    
    Returns:
        list: List of job dictionaries containing title, company, location, and description
    """
    logger.info(f"Starting job search for {query} with strict matching")
    
    def is_exact_match(job_title, search_query):
        """Check if job title closely matches the search query"""
        search_terms = search_query.lower().split()
        title_terms = job_title.lower().split()
        
        # Check if all search terms appear in title (order insensitive)
        return all(term in " ".join(title_terms) for term in search_terms)

    conn = http.client.HTTPSConnection("jobs-search-api.p.rapidapi.com")

    # If location is "India", use random cities
    if location.lower() == "india":
        logger.debug("Searching across multiple cities in India")
        jobs_per_city = max(1, results_wanted // len(INDIAN_CITIES))
        all_jobs = []

        for city in random.sample(INDIAN_CITIES, min(len(INDIAN_CITIES), results_wanted)):
            payload = json.dumps({
                "search_term": query,
                "location": f"{city}, India",
                "results_wanted": jobs_per_city * 2,  # Fetch extra to account for filtering
                "site_name": ["indeed", "linkedin", "zip_recruiter", "glassdoor"],
                "distance": 50,
                "job_type": "fulltime",
                "is_remote": False,
                "linkedin_fetch_description": True,
                "hours_old": 72,
            })

            headers = {
                "x-rapidapi-key": api_key,
                "x-rapidapi-host": "jobs-search-api.p.rapidapi.com",
                "Content-Type": "application/json",
            }

            try:
                logger.debug(f"Fetching jobs for city: {city}")
                conn.request("POST", "/getjobs", body=payload, headers=headers)
                res = conn.getresponse()
                data = res.read().decode("utf-8")
                city_jobs = json.loads(data).get("jobs", [])

                # Filter jobs for exact matches if strict_matching is True
                if strict_matching:
                    city_jobs = [job for job in city_jobs 
                               if is_exact_match(job["title"], query) and 
                               all(key in job for key in ["title", "company", "description"])]

                # Add city information to each job
                for job in city_jobs:
                    job["searched_location"] = city
                
                all_jobs.extend(city_jobs)

                # Stop if we've collected enough jobs
                if len(all_jobs) >= results_wanted:
                    logger.debug(f"Reached desired number of jobs: {results_wanted}")
                    break

            except Exception as e:
                logger.error(f"Error fetching jobs for {city}: {str(e)}")
                continue

        logger.info(f"Found {len(all_jobs)} matching jobs across Indian cities")
        return [{
            "job title": job["title"],
            "company": job["company"],
            "location": job.get("location", "N/A"),
            "searched_city": job.get("searched_location", "India"),
            "description": job["description"],
        } for job in all_jobs[:results_wanted]]

    else:
        # Original single-location logic with strict matching
        logger.debug(f"Searching in specific location: {location}")
        payload = json.dumps({
            "search_term": query,
            "location": location,
            "results_wanted": results_wanted * 2,  # Fetch extra to account for filtering
            "site_name": ["indeed", "linkedin", "zip_recruiter", "glassdoor"],
            "distance": 50,
            "job_type": "fulltime",
            "is_remote": False,
            "linkedin_fetch_description": True,
            "hours_old": 72,
            "show_requirements": True,
        })

        headers = {
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": "jobs-search-api.p.rapidapi.com",
            "Content-Type": "application/json",
        }
        
        try:
            conn.request("POST", "/getjobs", body=payload, headers=headers)
            res = conn.getresponse()
            data = res.read().decode("utf-8")
            job_data = json.loads(data)
            
            jobs = job_data.get("jobs", [])
            
            # Apply strict matching filter if enabled
            if strict_matching:
                jobs = [job for job in jobs 
                       if is_exact_match(job["title"], query) and 
                       all(key in job for key in ["title", "company", "description"])]

            return [{
                "job title": job["title"],
                "company": job["company"],
                "location": job.get("location", "N/A"),
                "searched_city": location.split(",")[0].strip(),
                "description": job["description"],
            } for job in jobs[:results_wanted]]
            
        except Exception as e:
            logger.error(f"Error fetching job: {str(e)}")
            return []



def clean_text(text):
    """
    Clean and normalize text by removing markdown formatting and excessive whitespace.

    Args:
        text (str): Input text to be cleaned

    Returns:
        str: Cleaned text
    """
    if not text:
        logger.debug("Recieved aempty text for cleaning")
        return ""

    logger.debug("Cleaning text content")
    text = re.sub(r"\*\*", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@st.cache_data(ttl=600) 
def documentation(job_details):
    """
    Convert the job details into LangChain Document format with cleaned text and metadata

    Args:
        job_details (List): List of job dictionaries from fetch_jobs()

    Returns:
        List: List of LangChain Document objects contains job description and metadata
    """
    logger.info("Converting jobs to Document format")
    content = []
    for job in job_details:
        try:
            doc = Document(
                page_content=clean_text(job["description"]),
                metadata={
                    "job_title": job["job title"],
                    "company": job["company"],
                    "location": job["location"],
                    "searched_city": job["searched_city"],
                    "language": "en",
                },
            )
            content.append(doc)

        except KeyError as e:
            logger.warning("Missing expected keys in data")
            continue

    logger.info("Document created")
    return content

@st.cache_data(ttl=600)
def fetch_jobs_strict(query, location="India", results_wanted=5, api_key=api_key, strict_matching=True):
    """
    Fetch job descriptions from the API based on query and location with strict role matching.
    
    Args:
        query (str): Job role for searching
        location (str, optional): Location to search. Defaults to "India".
        results_wanted (int, optional): Number of job results desired. Defaults to 5.
        api_key (str, optional): Rapid API key for authentication. Defaults to api_key.
        strict_matching (bool, optional): Whether to enforce strict job title matching. Defaults to True.
    
    Returns:
        list: List of job dictionaries containing title, company, location, and description
    """
    logger.info(f"Starting job search for {query} with strict matching")
    
    def is_exact_match(job_title, search_query):
        """Check if job title closely matches the search query"""
        search_terms = search_query.lower().split()
        title_terms = job_title.lower().split()
        
        # Check if all search terms appear in title (order insensitive)
        return all(term in " ".join(title_terms) for term in search_terms)

    conn = http.client.HTTPSConnection("jobs-search-api.p.rapidapi.com")

    # If location is "India", use random cities
    if location.lower() == "india":
        logger.debug("Searching across multiple cities in India")
        jobs_per_city = max(1, results_wanted // len(INDIAN_CITIES))
        all_jobs = []

        for city in random.sample(INDIAN_CITIES, min(len(INDIAN_CITIES), results_wanted)):
            payload = json.dumps({
                "search_term": query,
                "location": f"{city}, India",
                "results_wanted": jobs_per_city * 2,  # Fetch extra to account for filtering
                "site_name": ["indeed", "linkedin", "zip_recruiter", "glassdoor"],
                "distance": 50,
                "job_type": "fulltime",
                "is_remote": False,
                "linkedin_fetch_description": True,
                "hours_old": 72,
            })

            headers = {
                'x-rapidapi-key': api_key,
                'x-rapidapi-host': "jobs-search-api.p.rapidapi.com",
                'Content-Type': "application/json"
            }

            try:
                logger.debug(f"Fetching jobs for city: {city}")
                conn.request("POST", "/getjobs", body=payload, headers=headers)
                res = conn.getresponse()
                data = res.read().decode("utf-8")
                city_jobs = json.loads(data).get("jobs", [])

                # Filter jobs for exact matches if strict_matching is True
                if strict_matching:
                    city_jobs = [job for job in city_jobs 
                               if is_exact_match(job["title"], query) and 
                               all(key in job for key in ["title", "company", "description"])]

                # Add city information to each job
                for job in city_jobs:
                    job["searched_location"] = city
                
                all_jobs.extend(city_jobs)

                # Stop if we've collected enough jobs
                if len(all_jobs) >= results_wanted:
                    logger.debug(f"Reached desired number of jobs: {results_wanted}")
                    break

            except Exception as e:
                logger.error(f"Error fetching jobs for {city}: {str(e)}")
                continue

        logger.info(f"Found {len(all_jobs)} matching jobs across Indian cities")
        return [{
            "job title": job["title"],
            "company": job["company"],
            "location": job.get("location", "N/A"),
            "searched_city": job.get("searched_location", "India"),
            "description": job["description"],
        } for job in all_jobs[:results_wanted]]

    else:
        # Original single-location logic with strict matching
        logger.debug(f"Searching in specific location: {location}")
        payload = json.dumps({
            "search_term": query,
            "location": location,
            "results_wanted": results_wanted * 2,  # Fetch extra to account for filtering
            "site_name": ["indeed", "linkedin", "zip_recruiter", "glassdoor"],
            "distance": 50,
            "job_type": "fulltime",
            "is_remote": False,
            "linkedin_fetch_description": True,
            "hours_old": 72,
            "show_requirements": True,
        })

        headers = {
            'x-rapidapi-key': api_key,
            'x-rapidapi-host': "jobs-search-api.p.rapidapi.com",
            'Content-Type': "application/json"
        }
        
        try:
            conn.request("POST", "/getjobs", body=payload, headers=headers)
            res = conn.getresponse()
            data = res.read().decode("utf-8")
            job_data = json.loads(data)
            
            jobs = job_data.get("jobs", [])
            
            # Apply strict matching filter if enabled
            if strict_matching:
                jobs = [job for job in jobs 
                       if is_exact_match(job["title"], query) and 
                       all(key in job for key in ["title", "company", "description"])]

            return [{
                "job title": job["title"],
                "company": job["company"],
                "location": job.get("location", "N/A"),
                "searched_city": location.split(",")[0].strip(),
                "description": job["description"],
            } for job in jobs[:results_wanted]]
            
        except Exception as e:
            logger.error(f"Error fetching job: {str(e)}")
            return []