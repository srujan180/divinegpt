import os
import base64
from google.cloud import language_v1
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

credentials_base64 = os.environ.get("GOOGLE_CREDENTIALS_BASE64")
if credentials_base64:
    credentials_json = base64.b64decode(credentials_base64).decode("utf-8")
    with open("credentials.json", "w") as f:
        f.write(credentials_json)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"

client = language_v1.LanguageServiceClient()

# Langchain setup
API_KEY = os.environ.get("API_KEY")
llm = ChatGoogleGenerativeAI(
    apiKey=API_KEY,
    model="gemini-pro"
)

prompt_template = "Question: {question}\nAnswer:"
prompt = PromptTemplate.from_template(prompt_template)

chain = prompt | llm

st.title("DivineGPT")

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
user_question = st.text_input("Ask your Question:")

def handle_gpt(question):
    if question:
        with st.spinner('Thinking...'):
            result = chain.invoke({"question": question})  # Use invoke instead of run
            if isinstance(result, dict):  # Check if the response includes metadata
                answer = result.get("text", "No answer found")  # Extract the actual text
            else:
                answer = result
            st.session_state.conversation_history.append({"question": question, "answer": result})
            st.write(f"Answer: {result.content}")
    else:
        st.write("Please enter a question to get an answer.")

st.markdown("### Conversation History")
for entry in st.session_state.conversation_history:
    st.write(f"**Q:** {entry['question']}")
    st.write(f"**A:** {entry['answer']}")
    st.write("---")

st.markdown("### Example Questions")
st.write("- What is the capital of France?")
st.write("- Can you explain quantum computing in simple terms?")


handle_gpt(user_question)
