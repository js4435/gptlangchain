import os
import streamlit as st
from main import create_db_from_youtube_video_url, get_response_from_query, create_embeddings
import textwrap
import re

# 페이지 설정
st.set_page_config(page_title="YouTube Q&A App", layout="wide")

def convert_share_link_to_standard_url(url):
    if "youtu.be" in url:
        video_id = re.findall(r"youtu\.be/([^?/]+)", url)[0]
        return f"https://www.youtube.com/watch?v={video_id}"
    return url

# OpenAI API Key 입력 받기
if "api_key" not in st.session_state:
    st.session_state.api_key = None

if st.session_state.api_key is None:
    st.sidebar.subheader("API Key")
    api_key_input = st.sidebar.text_input("Enter your OpenAI API Key:")
    save_button = st.sidebar.button("Save API Key")

    if save_button:
        if api_key_input:
            st.session_state.api_key = api_key_input
            os.environ["OPENAI_API_KEY"] = api_key_input
            st.sidebar.success("API Key saved. You can use the app now.")
        else:
            st.sidebar.error("Please enter a valid API Key.")
else:
    st.sidebar.success("API Key is set. You can use the app now.")
    if st.sidebar.button("Reset API Key"):
        st.session_state.api_key = None
        del os.environ["OPENAI_API_KEY"]
        st.sidebar.success("API Key has been reset. Refresh the page to enter a new API Key.")

# 앱 제목과 설명
st.title("YouTube  Q&A App")
st.write("Ask a question based on a YouTube video, and get an answer!")

# 입력 필드
video_url = st.text_input("Enter YouTube video URL:")
video_url = convert_share_link_to_standard_url(video_url)
question = st.text_input("Enter your question:")

# 버튼을 누르면 쿼리 실행 및 결과 표시
if st.button("Get Answer"):
    if st.session_state.api_key:
        try:
            embeddings = create_embeddings(st.session_state.api_key) # OpenAI API Key를 이용하여 임베딩 생성
            db = create_db_from_youtube_video_url(video_url, embeddings) # YouTube 비디오 URL을 이용하여 데이터베이스 생성
            response, docs = get_response_from_query(db, question, embeddings) # 질문에 대한 답변 및 문서 검색
            st.write(textwrap.fill(response, width=100)) # 답변 출력
        except Exception as e:
            st.write(f"An error occurred: {e}") # 예외 처리
    else:
        st.write("Please enter and save your OpenAI API Key in the Settings.") # OpenAI API Key가 없는 경우 메시지 출력
