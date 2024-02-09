from customllm import CustomLLM
from embedder import CustomEmbedder
import time

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores.chroma import Chroma

# load the document 
loader = TextLoader("./state_of_the_union.txt")
documents = loader.load()

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
idList = []


# create the open-source embedding function
#embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


hf_embedder = HuggingFaceInferenceAPIEmbeddings(
    api_key="", api_url="", model_name="zephyr-7b-beta"
)

custom_embedder = CustomEmbedder()

# load it into Chroma
db = Chroma.from_documents(docs, embedding=custom_embedder)

time.sleep(30)

# query it
query = "Who was congratulated during the state of the union address of 2023?"
docs = db.similarity_search(query)

# print results
print(docs[0].page_content)


#question = "Who won the FIFA World Cup in the year 1994? "

#template = """Question: {question}

#Answer: Let's think step by step."""

#prompt = PromptTemplate(template=template, input_variables=["question"])

#llm = CustomLLM()
#llm_chain = LLMChain(prompt=prompt, llm=llm)

#print(llm_chain.invoke(question))