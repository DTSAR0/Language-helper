import streamlit as st
from app.core.prompts import WORD_PROMPT
from app.core.llm import call_llm

st.set_page_config(page_title="Language Helper (local)", page_icon="🗣️")
st.title("🗣️ Language Helper — locally, no keys required")

text = st.text_input("Enter a word or phrase")
col1, col2, col3 = st.columns(3)
src = col1.selectbox("Source language", ["en", "pl", "uk"])
tgt = col2.selectbox("Target language", ["uk", "pl", "en"])
model = col3.selectbox("Model (local)", ["qwen2.5:7b-instruct", "mistral", "llama3.1:8b-instruct"])

if st.button("Translate") and text:
    with st.spinner("Generating response locally..."):
        prompt = WORD_PROMPT.format(text=text, source=src, target=tgt)
        answer_md = call_llm(prompt, model=model)
    st.markdown(answer_md)
