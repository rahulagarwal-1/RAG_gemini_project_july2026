from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()

#api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL")

# New style
client = genai.Client()

myfile = client.files.upload(file="data/Kazhipathur 3.m4a")

print('-----response start-----')

response = client.models.generate_content(
    model=model_name,
    contents=["Describe this audio clip and tell me the tone of the speech.", myfile]
)
print(response.text)

print('-----response end-----')
