from custom_embedders import CustomChromaEmbedder
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma

if __name__ == "__main__":
    # load the document 
    loader = TextLoader("./state_of_the_union.txt")
    documents = loader.load()

    # split it into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # create the open-source embedding function
    embedding_function = CustomChromaEmbedder()

    # load it into Chroma
    db = Chroma.from_documents(documents=docs, embedding=embedding_function, persist_directory="./chromadb")
    print("embedding complete")