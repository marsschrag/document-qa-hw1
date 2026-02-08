import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
import requests

#title and description
st.title("Lab3 question answering chatbot")

st.write("""
         You can ask this chatbot anything and it will explain it in very simple terms. 
         You may provide up to 2 URLs for context, with a max character count of 50,000 per url.
         This chatbot will remember the past 6 messages (3 of yours and 3 it gives you).
         """)

#sidebar
st.sidebar.header("Configuration")

# URL inputs
url1 = st.sidebar.text_input("URL 1 (optional)", placeholder="https://example.com")
url2 = st.sidebar.text_input("URL 2 (optional)", placeholder="https://example.com")


llm_choice = st.sidebar.selectbox(
    "select LLM:",
    options=[
        "GPT-5-chat-latest (OpenAI)", 
        "Claude Sonnet 4.5 (Anthropic)"
    ]
)

# Function to read URL content
def read_url_content(url):
    """Fetch content from a URL"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {e}"
    
# Get URL content if provided
url_content = ""
if url1:
    content1 = read_url_content(url1)
    url_content += f"\n\n=== Content from URL 1 ===\n{content1}"
if url2:
    content2 = read_url_content(url2)
    url_content += f"\n\n=== Content from URL 2 ===\n{content2}"

max_chars = 50000
if len(url_content) > max_chars:
    url_content = url_content[:max_chars]

#create clients
if 'openai_client' not in st.session_state:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    st.session_state.openai_client = OpenAI(api_key=openai_api_key)

if 'anthropic_client' not in st.session_state:
    anthropic_api_key = st.secrets["CLAUDE_API_KEY"]
    st.session_state.anthropic_client = Anthropic(api_key=anthropic_api_key)

# Base system prompt
base_system_prompt = """you are a helpful assistant that explains in a way a 10-year-old can understand. 
ask 'Do you want more info?' after every question. 
If the user says yes, provide more detailed information and ask again. 
If the user says no, ask 'How else can I help you?'
Use the context provided from the URLs (if any) to answer questions."""

# Initialize or update system prompt
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = base_system_prompt + url_content
else:
    # Update system prompt if URLs changed
    st.session_state.system_prompt = base_system_prompt + url_content

# Initialize messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])

if prompt := st.chat_input("what is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    #6 previous messages
    if len(st.session_state.messages) > 6:
        st.session_state.messages = st.session_state.messages[-6:]



    # get the right llm model
    try:
        if "OpenAI" in llm_choice:
            client = st.session_state.openai_client
            stream = client.chat.completions.create(
                model="gpt-4o",
                messages=st.session_state.messages,
                stream=True
            )
            with st.chat_message("assistant"):
                response = st.write_stream(stream)
        else:  # Anthropic
            client = st.session_state.anthropic_client
            # Anthropic doesn't use system role in messages array
            system_content = st.session_state.messages[0]["content"]
            messages_without_system = [msg for msg in st.session_state.messages if msg["role"] != "system"]
            
            with client.messages.stream(
                model="claude-sonnet-4-5-20250929",
                max_tokens=2048,
                system=system_content,
                messages=messages_without_system
            ) as stream:
                with st.chat_message("assistant"):
                    response = st.write_stream(stream.text_stream)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    except Exception as e:
        st.error(f"Error: {e}")