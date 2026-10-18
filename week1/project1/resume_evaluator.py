import os
import time
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
from pydantic import BaseModel,Field

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")
if not my_api_key:
    raise ValueError("API KEY MISSING")

client=Groq(api_key=my_api_key)
model="openai/gpt-oss-20b"

job_description="""
We re seeking enthusiastic, early-career Front-End Developers to join our growing team. You ll work alongside experienced developers, contributing to the development and enhancement of our web and mobile applications. Your curiosity and eagerness to learn will be key as you dive into a variety of technologies and expand your skillset.

Responsibilities
Collaborate with design and development teams to create engaging and user-friendly interfaces.
Write clean, maintainable, and well-tested code in React, Node.js, and potentially other front-end frameworks.
Participate in code reviews to ensure quality and share knowledge.
Learn and apply new technologies to enhance our applications.
Stay updated on the latest trends and best practices in front-end development.
Qualifications
Bachelor s degree in Computer Science or related field, or equivalent experience.
0-3 years of experience with front-end technologies (React, Angular, Vue.js, etc.).
Solid understanding of HTML, CSS, and JavaScript.
Basic understanding of back-end technologies (Node.js, C#, VB.Net, etc.) is a plus, but not required.
A strong desire to learn, grow, and collaborate in a team environment.
Excellent problem-solving and communication skills.
Nice to Have
Experience with Power Apps or similar low-code platforms.
Familiarity with responsive design and cross-browser compatibility.
Contributions to open source projects or personal coding portfolio.
Role: Software Development - Other
Industry Type: IT Services & Consulting
Department: Engineering - Software & QA
Employment Type: Full Time, Permanent
Role Category: Software Development
Education
UG: Any Graduate
PG: Any Postgraduate
"""

class JobD(BaseModel):
    role:str
    required_skills:list[str]
    preferred_skills:list[str]
    minimum_experience:float | None
    education_requirements: list[str]
    responsibilities: list[str]

jobd_schema= JobD.model_json_schema()

system_prompt=f"""
You're an expert HR assistant. Your job is to read job descriptions and extract structured information from them. Return only valid JSON matching this schema: {jobd_schema}
IMPORTANT:
Do not return the schema itself.
do not return fields like "properties" ,"titles" or "type"
fill the schema with actual information extracted from job description.

if minimum experience is not mentioned, return  NULL.
if information for a list is missing return empty list.
DO  NOT INVENT INFORMATION.
"""


user_prompt= f"""
analyze the following job description {job_description}
"""

message_system={
    "role": "system",
    "content": system_prompt
}

message_user={
    "role":"user",
    "content": user_prompt
}

response_format={
    "type" : "json_object"
}

messages = [message_system , message_user]

response = client.chat.completions.create(model=model, messages=messages, response_format=response_format)
answer = response.choices[0].message.content

raw_json =answer

import json
job_data= json.loads(raw_json)
job = JobD(**job_data)

print(job.minimum_experience)
print(job.education_requirements)




#parsing real resumes:
class MatchResults(BaseModel):
    score:float
    details: dict


class Experience(BaseModel):
    company : str | None=None
    role: str | None=None
    duration: str| None=None
    description: str | None=None
    skills_used : list[str] = []

class Resume(BaseModel):
    name: str | None=None
    email: str | None=None
    phone :str | None=None
    total_experience_years: float | None=None

    skills: list[str] = []
    experiences: list[Experience] = []
    education : list[str] = []
    projects: list[str] = []
    certifications: list[str] = []

resume_schema = Resume.model_json_schema()


#making a function called "final_score" which is calling LLM every time..
def final_score(job,resume):
    match_schema=MatchResults.model_json_schema()
    prompt=f"""
    You are a HR recruiter.
    compare the candidate's resume with job description.

    job description:
    {job.model_dump_json(indent=2)}

    candidate resume:
    {resume.model_dump_json(indent=2)}

    return a JSON matching this schema: 
    {match_schema}

    give me:
    1. candidate name
    2. matching skills
    3. missing important skills
    4. whether experience requirement is met
    5. Overall match percentage from 0 to 100
    6. a short final verdict
    keep the response concise and easy to read
"""
    message={
        "role":"user",
        "content":prompt
    }

    messages=[message]
    
    response_format={
        "type":"json_object"
    }
    response= client.chat.completions.create(model=model, messages=messages, response_format=response_format)
    data=json.loads(response.choices[0].message.content)
    return MatchResults(**data)




#another function to parse resume info structurally from string -  calls LLM every time...
def parse_resume(resume_text):
    system_prompt = f"""
    You are an expert resume parser.
    Extract information from the resume based on its meaning,
    not only based on exact section headings.

    Different resumes may use different headings.
    For example:
    - Experience
    - Professional Experience
    - Work History
    - Employment
    - Internships

    These may all contain relevant experience.

    Skills may also appear in the skills section, work experience,
    internships or projects.

    Return ONLY valid JSON matching this schema:
    {resume_schema}

    Important rules:
    1. Do not invent information.
    2. If a value is not available, return null.
    3. If a list has no information, return an empty list.
    4. Include internships inside experiences.
    5. Extract skills mentioned across the entire resume.
    """

    user_prompt = f"""
    Parse the following resume:
    {resume_text}
    """
    message_system={
        "role" : "system",
        "content" : system_prompt
    }
    message_user={
        "role" : "user",
        "content" : user_prompt
    }
    
    messages=[message_system, message_user]
    response_format={
        "type": "json_object"
    }
    
    response=client.chat.completions.create(model=model, messages=messages, response_format=response_format)
    raw_output = response.choices[0].message.content
    data = json.loads(raw_output)
    resume = Resume(**data)
    return resume


#importing our pdf or Docx reader libraries from python-----------------

from pypdf import PdfReader
from docx import Document

#making a function to convert pdf info into a single string--------------
def read_pdf(file_path):
    reader=PdfReader(file_path)
    text=""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text



#making a function to convert docx info into a single string---------------
def read_docx(file_path):
    document = Document(file_path)
    text=""
    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            text += paragraph.text+"\n"

# to handle tabular data in Docx
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    text +=cell.text+"\n"
    return text

#function to decide where to send the resume, based on .pdf or .docx

def read_resume(file_path):
    if file_path.suffix.lower() == ".pdf":
        return read_pdf(file_path)
    elif file_path.suffix.lower() == ".docx":
        return read_docx(file_path)
    else:
        return None


#actual functioning-------------------
resume_folder = Path("resumes")
all_results=[]
for file_path in resume_folder.iterdir():
    if file_path.suffix.lower() not in [".pdf" , ".docx"]:
        continue
    print("\nProcessing:" , file_path.name)
    resume_text = read_resume(file_path)
    parsed_resume= parse_resume(resume_text)
    time.sleep(5) #using sleep sto prevent Rate Limiting of LLm - (used to handle DOS attack)- cuz "parse_resume" is calling the llm.

    result = final_score(job, parsed_resume)
    time.sleep(5) #same here, cuz "final_score" is doing LLM call too.

    print("Score:" , result.score)
    all_results.append({
        "name": parsed_resume.name,
        "score": result.score,
        "details": result.details
    })


#sorting the array based on score
all_results.sort(
    key = lambda candidate: candidate["score"],
    reverse=True
    )
top = all_results[:1] #storing top results
worst=all_results[-1:] #storing bottom results

#printing top 2 candidates for job role from all resumes

print("Top candidate")
for candidate in top:
    print(candidate["name"],"-",candidate["score"],"%")
    print(candidate["details"])


#printing worst 2 candidates  for the job role from all resumes
print("Lowest candidate")
for candidate in worst:
    print(candidate["name"],"-",candidate["score"],"%")
    print(candidate["details"])

