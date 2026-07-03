from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os 
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv()

print("Loading PDF into text...")
loader = PyPDFLoader("/Users/rahulagarwal/Documents/ai practice/data/Aurelius_Robotics_Corporation_ARC_Sample_RAG_Document.pdf")
text_data = loader.load()
print("PDF loaded into text.")


# create chunks of text data
print("Creating chunks of text data...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
text_chunks = text_splitter.split_documents(text_data)
print("Chunks of text data created.")
print(f"Number of text chunks created: {len(text_chunks)}")

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)

try:
    chroma_db = Chroma.from_documents(text_chunks, embedding_model, persist_directory="./vector_db")
except Exception as e:
    print(f"Error creating Chroma database: {e}")


chroma_db_con = Chroma(persist_directory="./vector_db", embedding_function=embedding_model)


llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL"),
    temperature=0
)

user_query = "what is the valuation of reliance industries?"
relevant_chunks = chroma_db_con.similarity_search(user_query, k=3)
relevant_chunks_content = []
for i,chunks in enumerate(relevant_chunks):
    relevant_chunks_content.append(chunks.page_content)
    print(f"Relevant chunk {i}: {chunks.page_content}")



relevant_chunks_content = str(relevant_chunks_content)
response = llm.invoke(f'{user_query},context={relevant_chunks_content}')
print(response.content[0]['text'])