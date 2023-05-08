import os
import streamlit as st
from main import create_db_from_youtube_video_url, get_response_from_query, create_embeddings
import textwrap

st.set_page_config(page_title="YouTube Transcript Q&A App", layout="wide")

# Side bar
if not os.environ.get("OPENAI_API_KEY"):
    st.sidebar.title("Settings")
    api_key = st.sidebar.text_input("Enter your OpenAI API Key:")

    if st.sidebar.button("Save API Key"):
        os.environ["OPENAI_API_KEY"] = api_key
else:
    st.sidebar.success("API Key saved. You can use the app now.")


# App title and description
st.title("YouTube Transcript Q&A App")
st.write("Ask a question based on a YouTube video transcript, and get an answer!")

# Input fields
video_url = st.text_input("Enter YouTube video URL:")
question = st.text_input("Enter your question:")

# Button to run the query and show the result
if st.button("Get Answer"):
    if os.environ.get("OPENAI_API_KEY"):
        try:
            embeddings = create_embeddings(os.environ["OPENAI_API_KEY"])
            db = create_db_from_youtube_video_url(video_url, embeddings)
            response, docs = get_response_from_query(db, question, embeddings)
            st.write(textwrap.fill(response, width=50))
        except Exception as e:
            st.write(f"An error occurred: {e}")
    else:
        st.write("Please enter and save your OpenAI API Key in the Settings.")
