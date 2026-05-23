## Script to define the State object used by various agents
from typing import TypedDict, List, Dict, Any

class ResumeState(TypedDict):
    """
    Single profile state information
    """
    raw_resume: str                  
    clean_resume: str                # Basic cleaned text using regex
    anonymized_resume: str           # LLM cleaned resume to remove personal data

    cv_id: int                       # Origin index mapping back to original dataframe row containing JD and CV pairs