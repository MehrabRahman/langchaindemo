from customllm import CustomLLM
from custom_embedders import CustomChromaEmbedder
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma

question = "What is the CHIPS Act?"
template = """Instructions: Use only the following context to answer the question.

Context: {context}
Question: {question}
"""

embedder = CustomChromaEmbedder()
db = Chroma(persist_directory="./chromadb", embedding_function=embedder)
retriever = db.as_retriever(search_kwargs={"k": 1})
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

llm = CustomLLM()
# chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print(chain.invoke(question))