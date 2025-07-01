import streamlit as st
import os


from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

## Langsmith Tracking
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

## Prompt Templates
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant please answer to the user's queries based on best of your knowledge."),
        ("user", "{question}")
    ]
)


def generate_response(question):
    parser = StrOutputParser()
    llm = ChatGroq(model = "deepseek-r1-distill-llama-70b")
    chain = prompt | llm | parser
    answer = chain.invoke(
        {"question": question}
    )
    return answer


## Title
st.title("Question and Answering with Groq - Deepseek")

## Sidebar title for description
st.sidebar.title("Write your question here:")
st.sidebar.text("This app uses Groq's Deepseek model to answer your questions. Enter your question in the text area below and click 'Submit' to get an answer. The model is designed to provide accurate and helpful responses based on the input provided.")

# Sidebar for user input
st.sidebar.header("User Input")
user_question = st.sidebar.text_area("Enter Your Question")

## Button to submit the question
if st.sidebar.button("Submit"):
    if user_question:
        with st.spinner("Generating response...."):
            try:
                response = generate_response(user_question)
                st.subheader("Response to your Question:")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.error("Please enter a question before submitting.")      



