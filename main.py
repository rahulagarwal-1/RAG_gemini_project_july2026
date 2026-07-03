from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()

#api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL")

# New style
client = genai.Client()
var = '1,2,3'
prompt = f"""
What is the sum of the following numbers: {var}?
what is the product of the following numbers: {var}?
what is the average of the following numbers: {var}?
whats special about them. 
"""
# no langchain used
response = client.models.generate_content(
    model=model_name,
    contents=prompt,
    config=types.GenerateContentConfig(
        system_instruction='''
        answer in bullet points all in capital letters. 
        maximum 25 words per bullet point. 
        no more than 4 bullet points.
        ''',
        temperature=0.3
        )
)

print(response.text)