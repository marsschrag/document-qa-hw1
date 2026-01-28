import streamlit as st
from openai import OpenAI

# Show title and description.
st.title("Lab2")
st.write(
    "Upload a document below and GPT will summarize it! "
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.secrets.OPENAI_API_KEY

    # Create an OpenAI client.
client = OpenAI(api_key=openai_api_key)

    # Let the user upload a file via `st.file_uploader`.
uploaded_file = st.file_uploader(
    "Upload a document (.txt or .md)", type=("txt", "md")
)

#have user select a summary option
summary_option = st.radio(
    "Choose your summary format:",
    options=[
        "Summarize in 100 words",
        "Summarize in 2 connecting paragraphs", 
        "Summarize in 5 bullet points"
    ],
    disabled=not uploaded_file,
)

#checkbox for advanced model
use_advanced = st.checkbox("Use advanced model", disabled=not uploaded_file)

if uploaded_file and summary_option:

    # Process the uploaded file and question.
    document = uploaded_file.read().decode()

    #give model the prompt
    if summary_option == "Summarize in 100 words":
        prompt = "Summarize this document in exactly 100 words."
    elif summary_option == "Summarize in 2 connecting paragraphs":
        prompt = "Summarize this document in 2 connecting paragraphs."
    else:
        prompt = "Summarize this document in 5 bullet points."
    
    messages = [
        {
            "role": "user",
            "content": f"Here's a document: {document} \n\n---\n\n {summary_option}",
        }
    ]

    #select model
    model = "gpt-5-nano" if use_advanced else "gpt-5-chat-latest"

    # Generate an answer using the OpenAI API.
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )

    # Stream the response to the app using `st.write_stream`.
    st.write_stream(stream)
