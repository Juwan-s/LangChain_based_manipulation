import os
from dotenv import load_dotenv

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_upstage import UpstageEmbeddings

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.5,
    openai_api_key=openai_api_key
)

embeddings = UpstageEmbeddings(model="solar-embedding-1-large")

# Prompt 템플릿 구성
prompt_template = PromptTemplate(
    input_variables=["user_input"],
    template="""
    사용자가 입력한 문장에서 특정 도구에 관련된 대상을 추출하세요.
    문장의 예: "{user_input}"
    출력 형식: <추출된 대상 단어>
    * 출력할 때는 대상 단어만 출력하세요. 절대 부연 설명이나 다른 내용을 출력하지 마세요.
    * 도구가 여러 개인 경우, 공백 하나로 구분하여 출력하세요.
    """
)

# 특정 사전 기반의 체인을 생성하는 함수입니다. 이 체인은 사전에 정의된 규칙에 따라 질문에 응답합니다.
def get_chain(llm):

# LLMChain 생성
    extract_chain = LLMChain(llm=llm, prompt=prompt_template)

    return extract_chain  # 사전 기반 체인 반환

# Word embedding dictionary (예시)
class_name = {
    "embedding_vector1": "driver",
    "embedding_vector2": "hammer",
    "embedding_vector3": "wrench"
}

# 사용자 입력 예시
user_input = input("문장을 입력하세요: ")

extract_chain = get_chain(llm)

# 대상 단어 추출
response = extract_chain.invoke({"user_input": user_input})  # .run() 대신 .invoke() 사용

print(f"response: {response}")


print(f"일단 response: {response['text']}")

temp = response['text'].strip().split()


print(f"temp: {temp}")


res = embeddings.embed_documents(temp)

print(f"res: {len(res), len(res[0])}")


# response = StrOutputParser().parse(response)
# 대상 단어 파싱

# get_embedding_vector



# if "대상 단어:" in response:
#     target_word = response.split("대상 단어:")[1].strip()
#     print(f"추출된 단어: {target_word}")
    
#     # embedding vector 검색
#     embedding_vector = get_embedding_vector(target_word, class_name)
#     if embedding_vector:
#         print(f"대상 단어 '{target_word}'에 대한 embedding vector: {embedding_vector}")
#     else:
#         print(f"대상 단어 '{target_word}'에 대한 embedding vector를 찾을 수 없습니다.")
# else:
#     print("대상 단어를 추출할 수 없습니다.")