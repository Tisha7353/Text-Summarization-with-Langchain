import os
from urllib.parse import urlparse

import streamlit as st
import validators
from langchain_community.document_loaders import UnstructuredURLLoader, YoutubeLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------- settings
# Rough character limits (about 4 characters per token). Content shorter than
# SINGLE_PASS_CHARS goes to the model in one request; longer content is split
# into chunks, each chunk is summarized, then the summaries are combined.
SINGLE_PASS_CHARS = 24_000
CHUNK_SIZE = 12_000
CHUNK_OVERLAP = 300

YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
}

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
    )
}

CHUNK_PROMPT = PromptTemplate.from_template(
    "Summarize the following part of a longer text. Keep every key point.\n\n"
    "Text:\n{text}\n\nSummary:"
)

FINAL_PROMPT = PromptTemplate.from_template(
    "Provide a summary of the following content in about {words} words.\n\n"
    "Content:\n{text}\n"
)


# ---------------------------------------------------------------- helpers
def is_youtube_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in YOUTUBE_HOSTS


def load_docs(url: str):
    """Load a YouTube transcript or a web page as LangChain documents."""
    if is_youtube_url(url):
        # add_video_info=True depends on pytube, which breaks often, so it is
        # left off. The transcript itself is all the summary needs.
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
    else:
        loader = UnstructuredURLLoader(
            urls=[url], ssl_verify=False, headers=BROWSER_HEADERS
        )
    return loader.load()


def summarize(llm: ChatGroq, docs, words: int) -> str:
    text = "\n\n".join(d.page_content for d in docs).strip()
    if not text:
        raise ValueError("No text could be extracted from this URL.")

    parser = StrOutputParser()
    final_chain = FINAL_PROMPT | llm | parser

    # Short content: a single request is enough.
    if len(text) <= SINGLE_PASS_CHARS:
        return final_chain.invoke({"text": text, "words": words})

    # Long content: summarize chunks first, then combine (map-reduce).
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_text(text)
    chunk_chain = CHUNK_PROMPT | llm | parser
    # Low concurrency keeps us under Groq's rate limits.
    partials = chunk_chain.batch(
        [{"text": c} for c in chunks], config={"max_concurrency": 2}
    )
    combined = "\n\n".join(partials)
    return final_chain.invoke({"text": combined, "words": words})


# ---------------------------------------------------------------- UI
st.set_page_config(
    page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜"
)
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")

with st.sidebar:
    groq_api_key = st.text_input(
        "Groq API Key", value=os.getenv("GROQ_API_KEY", ""), type="password"
    )
    # Groq retires models from time to time. If you get a "model not found"
    # error, pick a current one from https://console.groq.com/docs/models
    model_name = st.text_input("Groq model", value="llama-3.1-8b-instant")
    words = st.slider("Summary length (words)", 100, 600, 300, step=50)

generic_url = st.text_input("URL", label_visibility="collapsed")

if st.button("Summarize the Content from YT or Website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It may be a YouTube video URL or a website URL.")
    else:
        try:
            with st.spinner("Waiting..."):
                # Create the model only after the inputs are validated.
                llm = ChatGroq(model=model_name.strip(), groq_api_key=groq_api_key.strip())
                docs = load_docs(generic_url.strip())
                st.success(summarize(llm, docs, words))
        except Exception as e:
            st.error(f"Something went wrong: {e}")