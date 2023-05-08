from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import os
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import textwrap
from langchain.embeddings.openai import OpenAIEmbeddings

# OpenAI API key를 이용하여 OpenAIEmbeddings 객체 생성
def create_embeddings(api_key):
    return OpenAIEmbeddings(openai_api_key=api_key)

# Youtube video url을 이용하여 FAISS 객체 생성
def create_db_from_youtube_video_url(video_url, embeddings):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    # RecursiveCharacterTextSplitter를 이용하여 transcript를 chunk 단위로 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    # FAISS를 이용하여 document를 vector로 변환하여 저장
    db = FAISS.from_documents(docs, embeddings)
    return db

# query를 이용하여 유사한 document를 검색하고, ChatOpenAI를 이용하여 답변 생성
def get_response_from_query(db, query, embeddings, k=4):
    """
    gpt-3.5-turbo can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    # FAISS를 이용하여 query와 유사한 document 검색
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    # ChatOpenAI를 이용하여 답변 생성
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

    # 시스템 메시지 prompt template
    template = """
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed and in korean.
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # 사용자 질문 prompt template
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # ChatPromptTemplate을 이용하여 ChatOpenAI와 prompt를 연결
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    # LLMChain을 이용하여 ChatOpenAI와 prompt를 연결하여 답변 생성
    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs


