import streamlit as st
import random
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from transformers import T5Tokenizer, T5ForConditionalGeneration

# --- Load Model and Tokenizer ---
model_repo = "tomunizua/breast_cancer-qa_chatbot"
tokenizer = T5Tokenizer.from_pretrained(model_repo)
model = T5ForConditionalGeneration.from_pretrained(model_repo)


# --- Prediction Function ---
def generate_answer(question):
    input_text = f"question: {question}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
    output_ids = model.generate(
        input_ids, 
        max_length=512,
          num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,  # prevents repeating 3-grams
        repetition_penalty=1.5,  # discourages token reuse
        length_penalty=1.0,      # balances short vs. long answers
        temperature=0.7,         # adds a bit of randomness
        top_p=0.9,               # nucleus sampling
    )
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Fallback if model returns nothing or a repeat of the question
    if not answer.strip() or answer.strip().lower() == prompt.strip().lower():
        return "I'm sorry, I’m unable to answer that at the moment. Can I help with something else?"
    return answer

# --- Greeting Detection ---
def is_greeting(text):
    greetings = ["hi", "hello", "hey", "hiya", "good morning", "good afternoon", "good evening"]
    return text.strip().lower() in greetings

st.set_page_config(page_title="Breast Cancer Chatbot", page_icon="🎀")

# --- App Header ---
st.markdown(
    "<h1 style='text-align: center; color: #FF69B4;'>💖Breast Cancer Support ChatBot </h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; font-size: 18px; color: #C71585;'>Whether you're a patient, survivor, or just looking for information, I'm here to help you.</p>",
    unsafe_allow_html=True
)

# Sidebar content
with st.sidebar:
    st.header(" Chat Options")
    
    if st.button("🔄 Refresh Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("🔗 **Helpful Resources**", unsafe_allow_html=True)
    st.markdown("""
        <ul style="list-style-type: none; padding-left: 0;">
            <li>
                <a href="https://breastcancernow.org" target="_blank" style="color:#d63384; text-decoration:none;">
                     Breast Cancer Now UK
                </a>
            </li>
            <li>
                <a href="https://www.breastcancerafrica.org/" target="_blank" style="color:#d63384; text-decoration:none;">
                     Breast Cancer Initiative East Africa
                </a>
            </li>
        </ul>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("This chatbot is still in the developmental phase, and should be regarded for informational support only.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Initial greeting from chatbot
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hi there! I'm your Breast Cancer Support Pal. How can I assist you today? 💕"
    })

# Display chat history 
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input Field ---
if prompt := st.chat_input("Type your question here"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

     # Bot response
    with st.chat_message("assistant", avatar="🎀"):
        if is_greeting(prompt):
            response = "Hi there! I'm your Breast Cancer Support Pal. How can I assist you today? 💕"
        else:
            response = generate_answer(prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
