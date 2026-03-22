import streamlit as st
import anthropic
import pandas as pd
import io

st.set_page_config(
    page_title="News Monitoring Bot",
    layout="centered",
)

#constants
MODELS = {
    "claude-haiku-4-5-20251001": "Haiku 4.5 · Low cost",
    "claude-sonnet-4-5": "Sonnet 4.5 · Mid cost",
}
MAX_TOKENS = 1024

SYSTEM_PROMPT = """You are a news monitoring assistant for a large global law firm. \
Your job is to help lawyers and staff find relevant news about their clients and matters.

You have access to a curated set of news articles provided below. \
You must ONLY answer based on these articles — do not use outside knowledge or invent information.

RULES:
1. Only report on news from the provided articles. If something isn't covered, say so clearly.
2. When asked for "interesting", "top", or "important" news, rank articles by legal/business \
significance: regulatory actions, litigation, M&A, executive changes, financial distress, \
and major policy changes rank highest.
3. For each article you surface, provide: title, source, date, and a brief explanation of \
WHY it is notable from a legal or business perspective.
4. When searching by topic or company, use semantic matching — partial names, subsidiaries, \
and related entities count.
5. Be concise and professional. This is a law firm tool.
6. Format ranked lists as numbered items. Each item: bold headline, source/date on one line, \
then 1–2 sentences of legal context.

ARTICLE CORPUS:
{articles}"""


#rag helpers
def build_corpus(df: pd.DataFrame) -> str:
    """Serialize the dataframe into a structured text block for the system prompt."""
    rows = []
    for i, row in df.iterrows():
        fields = "\n".join(f"  {col}: {val}" for col, val in row.items() if pd.notna(val) and str(val).strip())
        rows.append(f"[Article {i + 1}]\n{fields}")
    return "\n\n".join(rows)


def detect_intent(text: str) -> str:
    """Classify the user query to apply the right prompt enhancement."""
    lower = text.lower()
    if any(kw in lower for kw in ["interesting", "top", "important", "significant", "notable", "highlight", "briefing"]):
        return "interesting"
    if any(kw in lower for kw in ["find", "search", "about", "news on", "tell me about"]):
        return "search"
    return "general"


def enhance_prompt(user_text: str) -> str:
    """Append RAG-style instructions based on detected intent."""
    intent = detect_intent(user_text)
    if intent == "interesting":
        return (
            user_text
            + "\n\n[INSTRUCTION: Return a numbered ranked list of the 5–8 most legally/commercially "
            "significant articles. For each: bold title, source + date, then 1–2 sentences on legal significance.]"
        )
    if intent == "search":
        return (
            user_text
            + "\n\n[INSTRUCTION: Search for all articles related to the query. List every match with "
            "title, source, date, and a relevance note. If none found, say so clearly.]"
        )
    return user_text


#build rag db once at startup
@st.cache_data(show_spinner="Building article index…")
def load_articles(file_bytes: bytes, filename: str) -> tuple[pd.DataFrame, str]:
    """Parse the uploaded CSV and build the in-memory corpus string."""
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.dropna(how="all")
    corpus = build_corpus(df)
    return df, corpus

def get_response(client, model_id, system, messages, enhanced_input):
    """Collect full streaming response and measure latency."""
    api_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
    api_messages.append({"role": "user", "content": enhanced_input})
    full_text = ""
    t0 = time.perf_counter()
    with client.messages.stream(
        model=model_id,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=api_messages,
    ) as stream:
        for text in stream.text_stream:
            full_text += text
            yield full_text, None  #stream chunk
    elapsed = time.perf_counter() - t0
    yield full_text, elapsed  #final chunk with timing


#sidebar
with st.sidebar:
    st.title("News Monitoring")
    st.caption("Law Firm Client Intelligence")
    st.divider()

    uploaded = st.file_uploader("Upload articles CSV", type=["csv"])

    if uploaded:
        df, corpus = load_articles(uploaded.read(), uploaded.name)
        st.success(f"{len(df)} articles loaded")
        with st.expander("Column preview"):
            st.write(list(df.columns))
        with st.expander("Sample articles (first 3)"):
            st.dataframe(df.head(3), use_container_width=True)
    else:
        corpus = None
        st.info("Upload a CSV to get started.\n\nExpected columns: `title`, `source`, `date`, `content` (or similar).")

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


#chat state
if "messages" not in st.session_state:
    st.session_state.messages = []

#main area
st.header("Client News Monitor", divider="gray")

if not uploaded:
    st.info("Upload a CSV in the sidebar to begin monitoring client news.")
else:
    #render conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    #chat input
    if user_input := st.chat_input("Ask about the news…"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        #build API request
        system = SYSTEM_PROMPT.format(articles=corpus)
        api_messages = []
        for m in st.session_state.messages[:-1]:  #history (exclude current)
            api_messages.append({"role": m["role"], "content": m["content"]})
        api_messages.append({"role": "user", "content": enhance_prompt(user_input)})

        #stream response
        client = anthropic.Anthropic(api_key=st.secrets.get("CLAUDE_API_KEY") or os.environ.get("CLAUDE_API_KEY"))
        with st.chat_message("assistant"):
            response_box = st.empty()
            full_response = ""
            with client.messages.stream(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=api_messages,
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    response_box.markdown(full_response + "▌")
            response_box.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})