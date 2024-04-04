import os
import pickle
from glob import glob

import pinecone
from dotenv import load_dotenv
from tqdm.auto import tqdm

from db.save_embeddings import PineconeDB
from pdf.read_pdf import get_pdf_content
from qa.generate_embeddings import OpenAIEmbeddings
import requests
import fitz

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")
INDEX_NAME = os.environ.get("INDEX_NAME")


pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

flatten = lambda lst: [item for sublist in lst for item in sublist]

openai_embeddings = OpenAIEmbeddings()

# Vector DB에 임베딩하여 저장시킬 PDF 파일 경로 입력 또는 PDF 경로의 URL 입력
# file_path = "/Users/parksangwoo/Documents/1.개발자료/5. AI/4. 기술자료/NIPS-2017-attention-is-all-you-need-Paper.pdf"
file_path = "/Users/parksangwoo/Documents/1.개발자료/5. AI/4. 기술자료/Orion14B_v3.pdf"

docs = ""
pdf_contents = []

if file_path.startswith("/"):
    docs = glob(file_path)
    for _, pdf in enumerate(tqdm(docs, desc="Getting PDFs")):
        pdf_contents.append(get_pdf_content(pdf))
elif file_path.startswith("h"):
    pdf_contents.append(get_pdf_content(file_path))

embeddings = openai_embeddings(flatten(pdf_contents), 512)
print(embeddings)

with open("./pdf_embeddings.pkl", "wb") as fp:
    pickle.dump(embeddings, fp)

batch_size = 50

db = PineconeDB(PINECONE_API_KEY, PINECONE_ENV, INDEX_NAME)
db(embeddings)
