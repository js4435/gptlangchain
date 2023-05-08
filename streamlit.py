import os
import streamlit as st
from main import create_db_from_youtube_video_url, get_response_from_query, create_embeddings
import textwrap

st.set_page_config(page_title="YouTube Transcript Q&A App", layout="wide")

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

# App title and description
st.title("YouTube Transcript Q&A App")
st.write("Ask a question based on a YouTube video transcript, and get an answer!")

# Input fields
video_url = st.text_input("Enter YouTube video URL:")
question = st.text_input("Enter your question:")

# Button to run the query and show the result
if st.button("Get Answer"):
    if st.session_state.api_key:
        try:
            embeddings = create_embeddings(st.session_state.api_key)
            db = create_db_from_youtube_video_url(video_url, embeddings)
            response, docs = get_response_from_query(db, question, embeddings)
            st.write(textwrap.fill(response, width=50))
        except Exception as e:
            st.write(f"An error occurred: {e}")
    else:
        st.write("Please enter and save your OpenAI API Key in the Settings.")
