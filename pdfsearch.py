from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from sentence_transformers import SentenceTransformer
# from langchain.vectorstores import Pinecone
# import pinecone
import textwrap
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "XXXXXXXXX" # API token from huggingface.co setting page

loader = PyPDFLoader("indianeconomy.pdf")
data = loader.load()

# print(document)

#Preprocessing


def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

# preprocessed_doc = wrap_text_preserve_newlines(str(document[0]))

# print(preprocessed_doc)

#Text Splitting to chunks

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
docs = text_splitter.split_documents(data)

print(len(docs))
print(docs[0])

#Text Embedding

embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(docs, embeddings)

#similarity search
query = "Who is the author of the article?"

doc = db.similarity_search(query)

print(wrap_text_preserve_newlines(str(doc[0].page_content)))
print("-----------")
#Q and A
llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.8, "max_length":512})

chain = load_qa_chain(llm, chain_type="stuff")

queryText = "What is the conclusion of the research?"

docsResult = db.similarity_search(query)
print(chain.run(input_documents = docsResult, question = queryText))

