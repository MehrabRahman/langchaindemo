from customllm import CustomLLM

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma


# load the document 
loader = TextLoader("./state_of_the_union.txt")
documents = loader.load()

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# load it into Chroma
db = Chroma.from_documents(docs, embedding_function)

# query it
query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)

# print results
print(docs[0].page_content)

# question = "Who won the FIFA World Cup in the year 1994? "
# context = "Portugal won the World Cup in 1994."

# template = """Instructions: Use only the following context to answer the question.

# Context: {context}
# Question: {question}

# Answer: The FIFA World Cup winner in 1994 is"""

# prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# llm = CustomLLM()
# llm_chain = LLMChain(prompt=prompt, llm=llm)

# print(llm_chain.invoke({"context": context, "question": question}))