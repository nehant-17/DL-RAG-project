from langchain_community.llms import Ollama
import streamlit as st 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#Creating my prompts 
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",'You are heplful assistant. please respond to the question asked'),
        ("user","Question:{question}")
    ]
)

#Streamlit framework 
st.title("Langchain demo with gemma2:9b")
input_text = st.text_input('What question do you have in mind')

#ollama model
llm = Ollama(model="gemma2:9b")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))