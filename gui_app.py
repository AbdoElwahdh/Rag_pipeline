import streamlit as st
from src.augmented_LLM import generate
# import json
# import os

st.set_page_config(page_title="RAG Q&A", layout="centered")
st.title("📚 Retrieval-Augmented Q&A")

# User inputs a question
user_question = st.text_input("Ask a question:")

if st.button("Submit") and user_question.strip() != "":

    with st.spinner("Retrieving context and generating answer..."):
        answer = generate(user_question)

    st.subheader("Answer:")
    formatted_answer = answer.strip().replace("\n", "<br>")
    st.markdown(formatted_answer, unsafe_allow_html=True)
