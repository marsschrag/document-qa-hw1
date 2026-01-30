import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
import requests

# Show title and description.
st.title("HW2")
st.write(
    "Upload a URL below and GPT will summarize it! "
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.secrets.OPENAI_API_KEY
claude_api_key = st.secrets.CLAUDE_API_KEY

    # Create an OpenAI client.
openai_client = OpenAI(api_key=openai_api_key)
anthropic_client = Anthropic(api_key=claude_api_key)

    # Let the user upload a file via `st.file_uploader`.
document_url = st.text_input(
    "enter document url:",
    placeholder="https://example.com"
)

#have user select a summary option
summary_option = st.radio(
    "choose your summary format:",
    options=[
        "Summarize in 100 words",
        "Summarize in 2 connecting paragraphs", 
        "Summarize in 5 bullet points"
    ],
    disabled=not document_url,
)

output_language = st.selectbox(
    "select output language:",
    options=["English", "French", "Spanish", "Italian"],
    disabled=not document_url,
)

#checkbox for advanced model
llm_choice = st.sidebar.selectbox(
    "select LLM:",
    options=[
        "GPT-5-nano (OpenAI)", 
        "GPT-5-chat-latest (OpenAI)(Advanced)", 
        "Claude Haiku 4.5 (Anthropic)",
        "Claude Sonnet 4.5 (Anthropic)(Advanced)"
    ],
    disabled=not document_url,
)

if document_url and summary_option:

    try:

        # Process the url and question.
        response = requests.get(document_url)
        response.raise_for_status()
        document = response.text

        #give model the prompt
        if summary_option == "Summarize in 100 words":
            prompt = f"Summarize this document in exactly 100 words in {output_language}."
        elif summary_option == "Summarize in 2 connecting paragraphs":
            prompt = f"Summarize this document in 2 connecting paragraphs in {output_language}."
        else:
            prompt = f"Summarize this document in 5 bullet points in {output_language}."
        
        full_prompt = f"Here's a document: {document} \n\n---\n\n {prompt}"

        #select model
                # Route to the appropriate LLM based on selection
        if "OpenAI" in llm_choice:
            # OpenAI models
            model_mapping = {
                "GPT-5-nano (OpenAI)": "gpt-5-nano",
                "GPT-5-chat-latest (OpenAI)(Advanced)": "gpt-5-chat-latest"
            }
            model = model_mapping[llm_choice]
            
            messages = [{"role": "user", "content": full_prompt}]
            stream = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )
            st.write_stream(stream)
            
        else:
            # Anthropic models
            model_mapping = {
                "Claude Haiku 4.5 (Anthropic)": "claude-haiku-4-5-20251001",
                "Claude Sonnet 4.5 (Anthropic)(Advanced)": "claude-sonnet-4-5-20250929"
            }
            model = model_mapping[llm_choice]
            
            # Anthropic streaming
            with anthropic_client.messages.stream(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": full_prompt}]
            ) as stream:
                st.write_stream(stream.text_stream)

    except requests.RequestException as e:
        print(f"error fetching {document_url}: {e}")
    except Exception as e:
        st.error(f"error generating summary: {e}")

