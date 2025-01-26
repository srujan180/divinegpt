import os
from google.cloud import language_v1
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Manually specify the credentials file path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\gen-lang-client-0363623531-40186c5c14f7.json"

# Initialize Google Cloud Language API Client
client = language_v1.LanguageServiceClient()

# Langchain setup
API_KEY = "AIzaSyDMGiysGIjS8q7KytlMbzzWU7m0rMLHIGM"
llm = ChatGoogleGenerativeAI(
    apiKey=API_KEY,
    model="gemini-pro"
)

# Define the prompt template
prompt_template = "Question: {question}\nAnswer:"
prompt = PromptTemplate.from_template(prompt_template)

# Set up the LLM Chain
chain = LLMChain(llm=llm, prompt=prompt)

# Streamlit app setup
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

# Handle GPT logic when user submits a question
handle_gpt(user_question)

