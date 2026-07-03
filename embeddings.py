from google import genai
from dotenv import load_dotenv
import os
import csv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL")

client = genai.Client()

result = client.models.embed_content(
        model="gemini-embedding-2",
        contents="What is the meaning of life?"
)

print(result.embeddings)
#save to csv


with open('embeddings.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(result.embeddings)