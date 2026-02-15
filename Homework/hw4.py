import streamlit as st
from openai import OpenAI
import sys

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from pathlib import Path
from PyPDF2 import PdfReader

openAI_model = st.sidebar.selectbox("which model?", 
                    ("mini", "regular"))
if openAI_model == "mini":
    model_to_use = "gpt-5-nano"
else:
    model_to_use = "gpt-5-chat-latest"

#create openai client
if 'openai_client' not in st.session_state:
    api_key = st.secrets["OPENAI_API_KEY"]
    st.session_state.openai_client = OpenAI(api_key=api_key)

def remove_html_tags(text):
    result = []
    inside_tag = False
    
    for char in text:
        if char == '<':
            inside_tag = True
        elif char == '>':
            inside_tag = False
        elif not inside_tag:
            result.append(char)
    
    return ''.join(result)

def extract_text_from_html(html_path):
    try:
        with open(html_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        
        #remove script tags and their content
        while '<script' in html_content.lower():
            start = html_content.lower().find('<script')
            end = html_content.lower().find('</script>', start)
            if end == -1:
                break
            html_content = html_content[:start] + html_content[end+9:]
        
        #remove style tags and their content
        while '<style' in html_content.lower():
            start = html_content.lower().find('<style')
            end = html_content.lower().find('</style>', start)
            if end == -1:
                break
            html_content = html_content[:start] + html_content[end+8:]
        
        #remove all HTML tags
        text = remove_html_tags(html_content)
        
        #clean up whitespace
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = ' '.join(line.split())
            if line:
                cleaned_lines.append(line)
        text = ' '.join(cleaned_lines)
        return text.strip()
        
    except Exception as e:
        st.error(f"Error reading {html_path}: {e}")
        return None

#I chunked the documents into halves because that felt the easiest. It divides everything into the beginning and end content for each page and does not make everything super complicated.
def chunk_document(text, filename):
    midpoint = len(text) // 2
    
    #look for good breakpoint
    search_start = max(0, midpoint - 100)
    search_end = min(len(text), midpoint + 100)
    
    best_break = midpoint
    for i in range(search_start, search_end):
        if i < len(text) - 1 and text[i] == '.' and text[i+1] == ' ':
            if abs(i - midpoint) < abs(best_break - midpoint):
                best_break = i + 1
    
    #create two chunks
    chunk1 = text[:best_break].strip()
    chunk2 = text[best_break:].strip()
    
    #create unique IDs for each chunk
    chunk1_id = f"{filename}_chunk1"
    chunk2_id = f"{filename}_chunk2"
    
    return [
        {"id": chunk1_id, "text": chunk1, "metadata": {"filename": filename, "chunk": 1}},
        {"id": chunk2_id, "text": chunk2, "metadata": {"filename": filename, "chunk": 2}}
    ]

def add_chunk_to_collection(collection, chunk_text, chunk_id, metadata):
    client = st.session_state.openai_client
    response = client.embeddings.create(
        input=chunk_text,
        model='text-embedding-3-small'
    )
    embedding = response.data[0].embedding
    
    collection.add(
        documents=[chunk_text],
        ids=[chunk_id],
        embeddings=[embedding],
        metadatas=[metadata]
    )

def load_html_to_collection(folder_path, collection):
    folder = Path(folder_path)
    html_files = list(folder.glob("*.html"))
    
    if not html_files:
        st.warning(f"No HTML files found in {folder_path}")
        return False
    
    status_text = st.empty()
    
    total_chunks = 0
    
    for idx, html_file in enumerate(html_files):
        status_text.text(f"Processing {html_file.name}...")
        
        #cxtract text from HTML
        text = extract_text_from_html(html_file)
        
        if text and len(text) > 100:  #only process if meaningful content
            #chunk the document into 2 pieces
            chunks = chunk_document(text, html_file.name)
            
            #add each chunk to the collection
            for chunk in chunks:
                add_chunk_to_collection(
                    collection, 
                    chunk["text"], 
                    chunk["id"], 
                    chunk["metadata"]
                )
                total_chunks += 1
            
            st.success(f"Added {html_file.name} (2 chunks)")
    
    status_text.empty()
    st.success(f"Loaded {len(html_files)} HTML files as {total_chunks} chunks")
    return True

def create_vectordb():
    chroma_client = chromadb.PersistentClient(path='./ChromaDB_HTML')
    collection = chroma_client.get_or_create_collection('HTMLCourseCollection')
    
    #only load HTML files if collection is empty
    if collection.count() == 0:
        st.info("Creating vector database from HTML files...")
        loaded = load_html_to_collection('./HTML-Data/', collection)
        if loaded:
            st.success(f"Vector database created with {collection.count()} chunks")
    else:
        st.info(f"Vector database already exists with {collection.count()} chunks")
    return collection

def search_vectordb(collection, query, top_k=3):
    client = st.session_state.openai_client
    
    response = client.embeddings.create(
        input=query,
        model='text-embedding-3-small'
    )
    query_embedding = response.data[0].embedding
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results

def get_conversation_history():
    all_messages = st.session_state.messages
    
    #keep only last 5 interactions
    max_messages = 10
    
    if len(all_messages) > max_messages:
        recent_messages = all_messages[-max_messages:]
    else:
        recent_messages = all_messages

    conversation_history = []
    for msg in recent_messages:
        conversation_history.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    return conversation_history

def create_rag_prompt_with_history(user_question, retrieved_docs, conversation_history):
    #build context
    context = ""
    for idx, (doc_text, doc_id, metadata) in enumerate(zip(
        retrieved_docs['documents'][0], 
        retrieved_docs['ids'][0],
        retrieved_docs['metadatas'][0]
    )):
        filename = metadata.get('filename', 'Unknown')
        chunk_num = metadata.get('chunk', '?')
        context += f"\n--- Source: {filename} (Part {chunk_num}) ---\n"
        context += doc_text[:2000]
        context += "\n"
    
    #build conversation history string
    history_str = ""
    if conversation_history:
        history_str = "\n\nRECENT CONVERSATION HISTORY:\n"
        for msg in conversation_history:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role_label}: {msg['content']}\n"
    
    prompt = f"""You are a helpful course information assistant. You have access to course materials and conversation history.

course content:
{context}
{history_str}

current user question: {user_question}

Provide a clear and helpful answer. Consider the conversation history for context. If you're using information from the course documents, explicitly mention which document(s) you're referencing. If the information isn't in the provided documents, say so clearly."""

    return prompt

def chat_with_rag(user_message, collection):
    try:
        #search vector database for relevant documents
        retrieved_docs = search_vectordb(collection, user_message, top_k=3)
        
        #get conversation history
        conversation_history = get_conversation_history()
        
        #create RAG prompt with retrieved context AND conversation history
        rag_prompt = create_rag_prompt_with_history(user_message, retrieved_docs, conversation_history)
        
        #build messages
        messages = [
            {"role": "system", "content": "You are a helpful course information assistant. Be clear about whether you're using information from the provided course documents or your general knowledge. Use conversation history for context when relevant."}
        ]
        
        #add conversation history
        messages.extend(conversation_history)
        
        #add current user message with RAG context
        messages.append({"role": "user", "content": rag_prompt})
        
        #send to ChatGPT
        client = st.session_state.openai_client
        response = client.chat.completions.create(
            model=model_to_use,
            messages=messages,
            max_completion_tokens=1000,
        )
        
        response_text = response.choices[0].message.content
        return response_text, retrieved_docs
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return f"Error: {str(e)}", None

#initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

#initialize vector database (once)
if 'HTML_VectorDB' not in st.session_state:
    st.session_state.HTML_VectorDB = create_vectordb()

collection = st.session_state.HTML_VectorDB

st.title("Course HTML Chatbot with RAG & Memory")
st.write("Ask me anything about the course materials! I remember our last 5 interactions.")

st.sidebar.header("Knowledge Base Status")
st.sidebar.success(f"{collection.count()} document chunks loaded")

st.header("Chat")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("View sources"):
                for source in message["sources"]:
                    st.write(f"• {source}")

#chat input
if prompt := st.chat_input("Ask about the course..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, retrieved_docs = chat_with_rag(prompt, collection)
            
            if response and retrieved_docs:
                st.markdown(response)
                
                sources = [f"{meta.get('filename', 'Unknown')} (Part {meta.get('chunk', '?')})" 
                          for meta in retrieved_docs['metadatas'][0]]
                
                with st.expander("View sources used"):
                    st.write("Document chunks referenced:")
                    for source in sources:
                        st.write(f"• {source}")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "sources": sources
                })
    
    st.rerun()
