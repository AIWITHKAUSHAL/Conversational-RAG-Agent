# Video explanation guide (about 4–5 minutes)

Record the running app and the relevant code. Keep your real API key hidden. This file is a script, not a recorded submission.

## 0:00–0:40 — Goal and setup

“This is Recall, a conversational retrieval-augmented generation agent. It answers from uploaded documents and remembers recent conversation context. The language model is accessed through Euri using the OpenAI-compatible client. The API key comes from an environment variable or a password input.”

Show the UI and load the Agentic AI sample. Briefly point out PDF, TXT, and Markdown support.

## 0:40–1:40 — Demonstrate memory

Ask “What is Agentic AI?” Then ask “What are its benefits?” Open the second answer's retrieval details. Explain how the standalone query resolves “its” to Agentic AI. Show the answer citations and original retrieved text. Ask “Explain those benefits in simple terms” to show another follow-up.

## 1:40–2:30 — Explain the pipeline in code

Open `rag.py`. Show `read_document`, `retrieve`, and `ConversationalRAG.ask`.

“Documents are split into overlapping chunks. BM25 scores passages by query-term relevance. It is lexical retrieval, so it does not use an embedding API. On a follow-up, a first model call rewrites the question using recent messages; a second call answers from the retrieved passages and conversation context. Document text is treated as evidence rather than instructions.”

## 2:30–3:15 — Persistence and isolation

Show `Store.save_turn`, `Store.history`, and the session selector.

“SQLite stores documents, messages, source passages, and resolved queries per conversation. The entire transcript persists, but only the latest six turns are sent to the model to bound context. A new conversation has separate memory and documents. Clearing memory keeps the documents.”

Start a new conversation, then return to the previous saved one. Refresh to show persistence.

## 3:15–4:00 — Grounding and limitations

Ask “What is the capital of Iceland?” Explain that this knowledge base does not contain the answer. Show the evidence limitation. Mention that BM25 can miss synonyms, scanned PDFs require OCR, and citations still require review.

## 4:00–4:40 — Tests and close

Run `python -m unittest discover -s tests -v` using the project environment. Point out follow-up rewriting, session isolation, persistence, failure handling, and UI coverage. Explain that test model responses are mocked, while the demonstrated app uses your real Euri account.

Before submitting, record the actual live conversation, verify the selected model works with your account, and submit the video alongside the project according to your course instructions.
