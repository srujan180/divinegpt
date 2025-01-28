import os
import base64
from google.cloud import language_v1
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

credentials_base64 = os.environ.get("GOOGLE_CREDENTIALS_BASE64")
if credentials_base64:
    credentials_json = base64.b64decode(credentials_base64).decode("utf-8")
    with open("credentials.json", "w") as f:
        f.write(credentials_json)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"

client = language_v1.LanguageServiceClient()

API_KEY = os.environ.get("API_KEY")
llm = ChatGoogleGenerativeAI(
    apiKey=API_KEY,
    model="gemini-pro"
)

prompt_template = "Question: {question}\nAnswer:"
prompt = PromptTemplate.from_template(prompt_template)

chain = LLMChain(llm=llm, prompt=prompt)

st.title("DivineGPT")

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
user_question = st.text_input("Ask your Question:")

def handle_gpt(question):
    if question:
        with st.spinner('Thinking...'):
            result = chain.run(question=question)  # Get the answer from the chain
            st.session_state.conversation_history.append({"question": question, "answer": result})
            st.write(f"Answer: {result}")
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8501))
    st.write(f"Running on port {port}")
    st.run(host="0.0.0.0", port=port)
