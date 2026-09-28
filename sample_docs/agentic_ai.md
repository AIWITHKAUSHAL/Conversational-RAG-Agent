# Agentic AI and conversational RAG

Agentic AI is an approach in which an AI system works toward a goal by choosing actions, using tools, observing results, and adapting its next steps. A typical loop is plan, act, observe, and revise. Tools can include document search, calculators, and external APIs.

The benefits of Agentic AI include automating multi-step tasks, adapting to new information, and coordinating tools to solve a problem. For example, a research assistant can search documents, compare findings, and prepare a cited answer. Important limitations include incorrect plans, unreliable tool results, and the need for human oversight before consequential actions.

Traditional RAG, or retrieval-augmented generation, retrieves relevant passages from a knowledge base and supplies them to a language model to ground its response. Its main steps are document loading, chunking, indexing, retrieval, and answer generation. RAG helps provide evidence but does not guarantee every generated claim is correct.

Conversational RAG adds conversation memory. A follow-up such as 'What are its benefits?' can be rewritten using the preceding topic, for example 'What are the benefits of Agentic AI?' The rewritten query retrieves relevant passages, and the answer uses those passages alongside recent conversation history.

Conversation memory stores user and assistant messages under a session identifier. Keeping sessions separate prevents unrelated conversations from sharing context. A bounded recent-message window limits the amount of history sent to the model. Persistent storage allows the conversation to be reopened after restarting the app.

BM25 is a lexical retrieval algorithm. It scores passages using matching query terms, term frequency, inverse document frequency, and document-length normalization. It does not require an embedding model, but it may miss synonyms that do not share words. Dense vector retrieval instead compares learned embeddings and can improve semantic matching.

Source citations let a reader inspect the passages used to generate an answer. When the documents do not contain enough information, a grounded assistant should acknowledge that limitation instead of inventing an answer.
