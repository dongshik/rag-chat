import os, sys
from time import sleep
import pinecone

# 상위 디렉토리의 db,pdf,qa 폴더 참조하기위해서
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
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
                 "available in the extracted text you don't answer it. Please translate it into Korean.")


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

# 질문 : what is tokenizer
# 응답 : 
'''Tokenizer는 LLM의 기본 구성 요소로, 여러 언어의 텍스트에서 텍스트 분포를 나타내면서 훈련에 유리한 어휘 크기를 유지해야 합니다. 
다중 언어 토크나이저의 경우, 통계적 방법을 사용하여 여러 언어의 텍스트에서 단어 수준 또는 서브워드 수준의 토큰을 생성하는 것이 일반적입니다. 
우리는 byte-pair encoding (BPE) 알고리즘을 활용하여 SentencePiece를 통해 구현합니다. 
우리의 설정은 99.99%의 문자 커버리지를 보장하며, 희귀한 문자는 기본적으로 UTF-8 바이트로 처리됩니다. 
우리는 훈련 코퍼스에서 다양한 텍스트 유형을 선별하여 다양한 말뭉치를 구축하고 훈련 데이터 분포와 일치시킵니다. 
이에는 영어, 중국어(간체 및 번체), 일본어, 한국어 등이 포함됩니다.'''

# 질문 : what is tokenizer?
# 응답 : 
'''
Tokenizer는 LLM의 기본 구성 요소로, 여러 언어의 텍스트에서 단어 수준 또는 서브워드 수준의 토큰을 생성하기 위해 통계적 방법을 일반적으로 사용합니다. 
우리는 byte-pair encoding (BPE) 알고리즘을 활용하여 tokenizer를 구현하였습니다. 
이를 통해 99.99%의 문자 커버리지를 보장하며, 희귀한 문자는 기본적으로 UTF-8 바이트로 처리됩니다. 
우리는 훈련 데이터 분포와 일치하고 다양한 말뭉치를 구축하기 위해 영어, 중국어(간체 및 번체), 일본어, 한국어 등 다양한 언어의 텍스트 유형을 포함한 많은 텍스트 유형을 사용합니다.
'''



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