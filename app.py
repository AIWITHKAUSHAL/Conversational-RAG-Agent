"""Run with: python -m streamlit run app.py"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

from rag import ConversationalRAG, EuriChat, Store

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
st.set_page_config(page_title="Recall | Conversational RAG", page_icon="💬", layout="wide")
store = Store(os.getenv("RAG_DB_PATH", str(ROOT / "data" / "rag.db")))
if "conversation" not in st.session_state:
    sessions = store.sessions()
    st.session_state.conversation = sessions[0]["id"] if sessions else store.new_session()


def show_evidence(sources, query):
    with st.expander("Sources & retrieval details"):
        st.caption("Standalone search query")
        st.write(query)
        for index, source in enumerate(sources, 1):
            st.text(f"[{index}] {source['source']} · page {source['page']} · chunk {source['number']}")
            st.text(source["text"])


with st.sidebar:
    st.title("💬 Recall")
    st.caption("Your documents. A conversation that remembers.")
    if st.button("＋ New conversation", use_container_width=True):
        st.session_state.conversation = store.new_session()
        st.rerun()
    sessions = store.sessions()
    labels = {s["id"]: s["title"] for s in sessions}
    selected = st.selectbox("Saved conversations", list(labels),
                            index=list(labels).index(st.session_state.conversation),
                            format_func=lambda value: labels[value])
    if selected != st.session_state.conversation:
        st.session_state.conversation = selected
        st.rerun()
    session = st.session_state.conversation
    st.divider()
    st.subheader("Knowledge base")
    uploads = st.file_uploader("Add PDF, TXT, or Markdown", type=["pdf", "txt", "md"],
                               accept_multiple_files=True, key=f"files_{session}")
    if st.button("Index uploaded documents", disabled=not uploads, use_container_width=True):
        for upload in uploads:
            try:
                added = store.add_document(session, upload.name, upload.getvalue())
                st.success(f"{upload.name}: {'indexed' if added else 'already indexed'}")
            except ValueError as exc:
                st.error(str(exc))
    if st.button("Load sample: Agentic AI", use_container_width=True):
        sample = ROOT / "sample_docs" / "agentic_ai.md"
        store.add_document(session, sample.name, sample.read_bytes())
    documents = store.documents(session)
    st.caption(f"{len(documents)} documents · {len(store.chunks(session))} passages")
    for document in documents:
        st.text("• " + document["name"])
    st.divider()
    with st.expander("Euri settings", expanded=not bool(os.getenv("EURI_API_KEY"))):
        entered_key = st.text_input("Euri API key", type="password", help="Or set EURI_API_KEY in your local .env file.")
        api_key = entered_key.strip() or os.getenv("EURI_API_KEY", "")
        model = st.text_input("Model", value=os.getenv("EURI_MODEL", "gemini-3.5-flash-lite"))
        top_k = st.slider("Passages per answer", 1, 6, 4)
        st.caption("Document excerpts and recent chat history are sent to Euri when you ask a question.")
    if st.button("Clear conversation memory", use_container_width=True):
        store.clear_history(session)
        st.rerun()
    st.caption("Local, single-user app · chats and documents saved on this computer.")

st.title("Talk to your knowledge")
st.caption("Ask a question. Follow up naturally. Trace the answer to its source.")
history = store.history(session)
if not history:
    st.info("Load the Agentic AI sample or upload your documents, then add your Euri API key to start.")
    a, b, c = st.columns(3)
    a.markdown("**01 · Ask**\n\nWhat is Agentic AI?")
    b.markdown("**02 · Follow up**\n\nWhat are its benefits?")
    c.markdown("**03 · Inspect**\n\nOpen sources to see the evidence.")
for message in history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            show_evidence(json.loads(message["sources"]), message["query"])
if history:
    st.download_button("Download conversation", json.dumps(history, indent=2, ensure_ascii=False),
                       file_name="conversation.json", mime="application/json")

question = st.chat_input("Ask about your documents…", disabled=not documents)
if question:
    if not api_key or api_key == "your_euri_api_key_here":
        st.error("Enter your Euri API key in the sidebar or set EURI_API_KEY in .env.")
    elif not model.strip():
        st.error("Enter a model name in Euri settings.")
    else:
        client = None
        try:
            with st.spinner("Reading context and finding evidence…"):
                client = EuriChat(api_key, model.strip(), os.getenv("EURI_BASE_URL", "https://api.euron.one/api/v1/euri"))
                ConversationalRAG(store, client).ask(session, question, top_k)
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            # Provider exception bodies may include sensitive information; don't render them.
            from openai import AuthenticationError, APIConnectionError, RateLimitError
            if isinstance(exc, AuthenticationError):
                st.error("Euri rejected the API key. Check your key and account access.")
            elif isinstance(exc, RateLimitError):
                st.error("Euri's rate or quota limit was reached. Check your balance or try again later.")
            elif isinstance(exc, APIConnectionError):
                st.error("Could not reach Euri. Check your connection and try again.")
            else:
                st.error("The request failed. Check that your Euri account supports the selected model, then retry.")
        else:
            st.rerun()
        finally:
            if client is not None:
                client.close()
