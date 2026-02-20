import streamlit as st
from openai import OpenAI
import sys

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from pathlib import Path
from PyPDF2 import PdfReader

#create chromadb client
chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_Lab')
collection = chroma_client.get_or_create_collection('HW5Collection')

model_to_use = "gpt-5-chat-latest"

#create openai client
if 'openai_client' not in st.session_state:
    api_key = st.secrets["OPENAI_API_KEY"]
    st.session_state.openai_client = OpenAI(api_key=api_key)

def add_to_collection(collection, text, file_name):
    #create embedding
    client = st.session_state.openai_client
    response = client.embeddings.create(
        input=text,
        model='text-embedding-3-small'
    )
    #get embedding
    embedding = response.data[0].embedding
    
    #add embedding and doc to chromadb
    collection.add(
        documents=[text],
        ids=[file_name],
        embeddings=[embedding],
        metadatas=[{"filename": file_name}]
    )

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading {pdf_path}: {e}")
        return None


def load_pdfs_to_collection(folder_path, collection):
    folder = Path(folder_path)
    pdf_files = list(folder.glob("*.pdf"))
    
    if not pdf_files:
        st.warning(f"No PDF files found in {folder_path}")
        return False
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, pdf_file in enumerate(pdf_files):
        status_text.text(f"Processing {pdf_file.name}...")
        
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_file)
        
        if text:
            # Add to collection
            add_to_collection(collection, text, pdf_file.name)
            st.success(f"✓ Added {pdf_file.name}")
        
        # Update progress
        progress_bar.progress((idx + 1) / len(pdf_files))
    
    status_text.text("All PDFs processed!")
    return True
    
def create_vectordb():
    #create chromadb client
    chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_Lab')
    collection = chroma_client.get_or_create_collection('HW4Collection')
    
    #only load PDFs if collection is empty
    if collection.count() == 0:
        st.info("Loading PDFs into ChromaDB...")
        loaded = load_pdfs_to_collection('./HW-05-Data/', collection)
        if loaded:
            st.success(f"Successfully loaded {collection.count()} documents into ChromaDB!")
    else:
        st.info(f"ChromaDB already contains {collection.count()} documents")
    
    return collection

def search_vectordb(collection, query, top_k=3):
    client = st.session_state.openai_client
    
    #create embedding for the query
    response = client.embeddings.create(
        input=query,
        model='text-embedding-3-small'
    )
    query_embedding = response.data[0].embedding
    
    #query chromadb
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    return results

if collection.count() == 0:
    loaded = load_pdfs_to_collection('./HW-05-Data/', collection)

#title and description
st.title("Enhanced Chatbot Using RAG")


#for testing
#topic = st.sidebar.text_input('Topic', placeholder='type your topic here')
#if topic:
    #client = st.session_state.openai_client
    #response = client.embeddings.create(
        #input=topic,
        #model='text-embedding-3-small')
    #query_embedding = response.data[0].embedding
    #results = collection.query(
        #query_embeddings=[query_embedding],
        #n_results=3
    #)
    #st.subheader(f"results for {topic}")
    #for i in range(len(results['documents'][0])):
        #doc = results['documents'][0][1]
        #doc_id = results['ids'][0][1]
        #st.write(f"**{1+1}. {doc_id}")
#else:
    #st.info("enter a topic in the sidebar to search the collection.")
#end testing
    


def create_rag_prompt(user_question, retrieved_docs):
    context = ""
    for idx, (doc_text, doc_id) in enumerate(zip(retrieved_docs['documents'][0], retrieved_docs['ids'][0])):
        context += f"\n--- Document {idx+1}: {doc_id} ---\n"
        context += doc_text[:2000]  #limit each doc to 2000 chars to manage token limits
        context += "\n"
    
    #create the RAG prompt
    prompt = f"""You are a helpful course information assistant. You have access to course materials and documents.
Based on the following course documents, please answer the user's question. If the answer is found in the documents, cite which document(s) you're referencing. If the information is not in the provided documents, clearly state that you don't have that information in the course materials.
COURSE DOCUMENTS:
{context}

USER QUESTION: {user_question}

Please provide a clear and helpful answer. If you're using information from the course documents, explicitly mention which document(s) you're referencing. If the information isn't in the provided documents, say so clearly."""

    return prompt

def chat_with_rag(user_message, collection):
    # Use the helper which performs vector search and then calls the LLM
    response_text, retrieved_docs = relevant_course_info(user_message, collection, top_k=3)
    return response_text, retrieved_docs


def relevant_course_info(query, collection, top_k=3):
    client = st.session_state.openai_client

    # create embedding for the query
    response = client.embeddings.create(
        input=query,
        model='text-embedding-3-small'
    )
    query_embedding = response.data[0].embedding

    # query chromadb
    retrieved_docs = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    #build context from retrieved documents (limit to keep tokens reasonable)
    context = ""
    for idx, (doc_text, doc_id) in enumerate(zip(retrieved_docs['documents'][0], retrieved_docs['ids'][0])):
        context += f"\n--- Document {idx+1}: {doc_id} ---\n"
        context += doc_text[:2000] + "\n"

    #put retrieved docs into the system prompt so the LLM cannot call the function
    system_content = (
        "You are a helpful course information assistant. Use ONLY the following provided course documents when answering the user's question. If the answer is not present in the documents, state that the information is not available in the course materials."
        f"COURSE DOCUMENTS:{context}"
    )

    #call LLM with docs in the system prompt
    chat_response = client.chat.completions.create(
        model=model_to_use,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": query}
        ],
        max_completion_tokens=1000
    )

    return chat_response.choices[0].message.content, retrieved_docs

#initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []
  
#initialize vector database (only once)
if 'HW5_VectorDB' not in st.session_state:
    st.session_state.HW5_VectorDB = create_vectordb()

collection = st.session_state.HW5_VectorDB

#sidebar info
st.sidebar.header("Knowledge Base Status")
st.sidebar.success(f"{collection.count()} documents loaded")

#show document list
with st.sidebar.expander("View loaded documents"):
    all_docs = collection.get()
    if all_docs['ids']:
        for doc_id in all_docs['ids']:
            st.write(f"• {doc_id}")

#chat interface
st.header("Chat")

#display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        #show sources if available
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("view sources"):
                for source in message["sources"]:
                    st.write(f"• {source}")

#chat input
if prompt := st.chat_input("Ask about the course..."):
    #add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    #display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    #get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, retrieved_docs = chat_with_rag(prompt, collection)
            st.markdown(response)
            
            #show which documents were referenced
            sources = retrieved_docs['ids'][0]
            with st.expander("View sources used"):
                st.write("Documents referenced:")
                for source in sources:
                    st.write(f"• {source}")
    
    #add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response,
        "sources": sources
    })

    st.rerun()
