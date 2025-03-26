import http.client
import json
import random
import re
from langchain.schema import Document
from dotenv import load_dotenv
import os 

load_dotenv()

api_key=os.getenv('RAPIDAPI_KEY')

INDIAN_CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Ahmedabad",
    "Chennai", "Kolkata", "Surat", "Pune", "Jaipur",
    "Lucknow", "Kanpur", "Nagpur", "Visakhapatnam", "Indore",
    "Thane", "Bhopal", "Patna", "Vadodara", "Ghaziabad"
]

def fetch_jobs(query, location="India", results_wanted=5,api_key=api_key):
    conn = http.client.HTTPSConnection("jobs-search-api.p.rapidapi.com")
    
    # If location is "India", use random cities
    if location.lower() == "india":
        # Calculate how many jobs per city (at least 1 city per job)
        jobs_per_city = max(1, results_wanted // len(INDIAN_CITIES))
        all_jobs = []
        
        for city in random.sample(INDIAN_CITIES, min(len(INDIAN_CITIES), results_wanted)):
            payload = json.dumps({
                "search_term": query,
                "location": f"{city}, India",
                "results_wanted": jobs_per_city,
                "site_name": ["indeed", "linkedin", "zip_recruiter", "glassdoor"],
                "distance": 50,
                "job_type": "fulltime",
                "is_remote": False,
                "linkedin_fetch_description": True,
                "hours_old": 72
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
                city_jobs = json.loads(data).get("jobs", [])
                
                # Add city information to each job
                for job in city_jobs:
                    job["searched_location"] = city
                all_jobs.extend(city_jobs)
                
                # Stop if we've collected enough jobs
                if len(all_jobs) >= results_wanted:
                    break
                    
            except Exception as e:
                print(f"Error fetching jobs for {city}: {str(e)}")
                continue
                
        # Trim to exact result count and format
        return [
            {
                "job title": job["title"],
                "company": job["company"],
                "location": job.get("location", "N/A"),
                "searched_city": job.get("searched_location", "India"),
                "description": job["description"]
            }
            for job in all_jobs[:results_wanted]
            if all(key in job for key in ["title", "company", "description"])
        ]
    
    else:
        # Original single-location logic
        payload = json.dumps({
            "search_term": query,
            "location": location,
            "results_wanted": results_wanted,
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

        conn.request("POST", "/getjobs", body=payload, headers=headers)
        res = conn.getresponse()
        data = res.read().decode("utf-8")
        job_data = json.loads(data)

        return [
            {
                "job title": job["title"],
                "company": job["company"],
                "location": job.get("location", "N/A"),
                "searched_city": location.split(",")[0].strip(),
                "description": job["description"]
            }
            for job in job_data.get("jobs", [])
            if all(key in job for key in ["title", "company", "description"])
        ]
        
def clean_text(text):
    """Remove excessive newlines and markdown bold syntax"""
    text = re.sub(r'\*\*', '', text)  # Remove **bold** markers
    text = re.sub(r'\n{3,}', '\n\n', text)  # Replace 3+ newlines with double newlines
    return text.strip()

def documentation(job_details):
    content=[] 
    for job in job_details: 
        doc = Document(
                page_content=clean_text(job["description"]),
                metadata={
                    "job_title": job["job title"],
                    "company": job["company"],
                    "location": job["location"],
                    "searched_city": job["searched_city"],
                    
                    "language": "en"
                    }
                )
        content.append(doc)
    return content
        
        