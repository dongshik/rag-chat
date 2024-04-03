import os
from time import sleep
import pinecone
from qa.completion import Completion
from qa.embedding_request import generate_embeddings

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")
INDEX_NAME = os.environ.get("INDEX_NAME")
OPENAI_CHAT_COMPLETION_MODEL = os.environ.get("OPENAI_CHAT_COMPLETION_MODEL")
PINECONE_NAMESPACE = os.environ.get("PINECONE_NAMESPACE")


def clear_terminal():
    os.system("cls" if os.name == "nt" else "clear")


SYSTEM_PROMPT = ("You are a helpful AI assistant. You have to use the provided extracted text chunks in triple "
                 "backticks and answer the user query/question provided in single backticks. If the answer is not "
                 "available in the extracted text you don't answer it. Please answer in Korean")


class ChatPDF:
    def __init__(self, pinecone_api_key, pinecone_env, pinecone_index, pinecone_namespace):
        self.complete = Completion()
        self.pc = pinecone.Pinecone(api_key=pinecone_api_key, environment=pinecone_env)
        self.namespace = pinecone_namespace
        self.index = pinecone.Index(host=self.pc.describe_index(pinecone_index).get('host'), api_key=pinecone_api_key)

    def __get_embedding__(self, question: str):
        embeddings = generate_embeddings([{"text": question, "src": "", "page_no": ""}])
        embedding = embeddings[0]
        embedding = embedding.get("embedding")
        return embedding

    def __search__(self, question: str):
        embedding = self.__get_embedding__(question)
        sim = self.index.query(vector=[embedding], top_k=5, include_metadata=True)
        print(sim)
        texts = list(map(lambda s: s.get("metadata").get("text"), sim.get("matches")))
        return texts

    def __call__(self, message: str):
        search_results = self.__search__(message)
        message = f"""Extracted Text: ```{" ".join(search_results)}``` Question: `{message}`"""

        answer = ""
        for msg in self.complete(message, SYSTEM_PROMPT, OPENAI_CHAT_COMPLETION_MODEL):
            answer += msg
            clear_terminal()
            print(answer, end="", flush=True)
            sleep(0.1)


if __name__ == "__main__":
    cp = ChatPDF(PINECONE_API_KEY, PINECONE_ENV, INDEX_NAME, PINECONE_NAMESPACE)
    while True:
        question = input("\nEnter your question: ")
        question = question.rstrip().lstrip()
        if question:
            cp(question)
        else:
            print("Please provide a question or query!")
            pass