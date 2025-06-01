
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from load_and_embed import load_vectorstore
import streamlit as st

st.title("📚 書籍ベースAIチャットボット")

question = st.text_input("質問を入力してください：")

if question:
    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(question)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents=docs, question=question)
    st.write("🧠 回答:", response)
