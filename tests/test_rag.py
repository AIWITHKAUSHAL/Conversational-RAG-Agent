import json
from io import BytesIO
from pathlib import Path
import tempfile
import unittest

from rag import Chunk, ConversationalRAG, EuriChat, Store, read_document, retrieve


class FakeLLM:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def complete(self, messages, max_tokens=1000):
        self.calls.append(messages)
        response = next(self.replies)
        if isinstance(response, Exception):
            raise response
        return response


class RAGTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "test.db"
        self.store = Store(self.path)
        self.session = self.store.new_session()
        self.store.add_document(self.session, "agents.txt", b"Agentic AI uses tools and planning. Benefits include automating tasks.")

    def tearDown(self):
        self.temp.cleanup()

    def test_followup_uses_history_and_persists_sources(self):
        llm = FakeLLM(["Agentic AI uses tools [1].", "Agentic AI benefits", "It automates tasks [1]."])
        agent = ConversationalRAG(self.store, llm)
        agent.ask(self.session, "What is Agentic AI?")
        result = agent.ask(self.session, "What are its benefits?")
        self.assertEqual(result["query"], "Agentic AI benefits")
        self.assertEqual(llm.calls[1][1]["content"], "What is Agentic AI?")
        self.assertIn("automating tasks", llm.calls[2][0]["content"])
        restored = Store(self.path).history(self.session)
        self.assertEqual(len(restored), 4)
        self.assertEqual(json.loads(restored[-1]["sources"])[0]["source"], "agents.txt")

    def test_sessions_are_isolated(self):
        other = self.store.new_session()
        agent = ConversationalRAG(self.store, FakeLLM(["Uses tools [1]."]))
        agent.ask(self.session, "Agentic AI")
        self.assertEqual(self.store.history(other), [])
        self.assertEqual(self.store.chunks(other), [])
        with self.assertRaises(ValueError):
            agent.ask(other, "Agentic AI")

    def test_no_evidence_does_not_call_generation(self):
        llm = FakeLLM([])
        answer = ConversationalRAG(self.store, llm).ask(self.session, "quasars astronomy")
        self.assertEqual(answer["sources"], [])
        self.assertIn("couldn't find", answer["answer"])
        self.assertEqual(llm.calls, [])

    def test_failed_provider_call_does_not_save_partial_turn(self):
        agent = ConversationalRAG(self.store, FakeLLM([RuntimeError("offline")]))
        with self.assertRaises(RuntimeError):
            agent.ask(self.session, "Agentic AI")
        self.assertEqual(self.store.history(self.session), [])

    def test_clear_memory_preserves_documents(self):
        ConversationalRAG(self.store, FakeLLM(["Tools [1]."])).ask(self.session, "Agentic AI")
        self.store.clear_history(self.session)
        self.assertEqual(self.store.history(self.session), [])
        self.assertTrue(self.store.chunks(self.session))

    def test_duplicate_document_is_not_indexed_twice(self):
        self.assertFalse(self.store.add_document(self.session, "copy.txt", b"Agentic AI uses tools and planning. Benefits include automating tasks."))
        self.assertEqual(len(self.store.documents(self.session)), 1)

    def test_chunk_overlap_and_no_missing_words(self):
        words = [f"word{i}" for i in range(500)]
        chunks = read_document("long.txt", " ".join(words).encode())
        self.assertEqual(chunks[0].text.split()[-40:], chunks[1].text.split()[:40])
        self.assertEqual(set(" ".join(c.text for c in chunks).split()), set(words))

    def test_retrieval_ranks_relevant_passage_first(self):
        chunks = [Chunk("a", 1, 1, "Oranges grow on trees."), Chunk("b", 1, 1, "Agentic AI planning tools")]
        self.assertEqual(retrieve("Agentic AI", chunks)[0].source, "b")

    def test_invalid_documents_and_questions(self):
        for name, content in [("a.txt", b" "), ("a.exe", b"text"), ("a.txt", b"\xff")]:
            with self.assertRaises(ValueError):
                read_document(name, content)
        with self.assertRaises(ValueError):
            ConversationalRAG(self.store, FakeLLM([])).ask(self.session, " ")

    def test_missing_key_is_rejected(self):
        with self.assertRaises(ValueError):
            EuriChat("", "test", "https://example.com")

    def test_pdf_text_and_page_metadata(self):
        from pypdf import PdfWriter
        from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
        writer = PdfWriter()
        page = writer.add_blank_page(width=300, height=300)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                 NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 12 Tf 20 200 Td (Agentic AI uses tools.) Tj ET")
        page[NameObject("/Contents")] = stream
        output = BytesIO()
        writer.write(output)
        chunks = read_document("sample.pdf", output.getvalue())
        self.assertIn("Agentic AI", chunks[0].text)
        self.assertEqual(chunks[0].page, 1)

    def test_euri_request_contract(self):
        import httpx
        from openai import OpenAI
        def respond(request):
            self.assertEqual(str(request.url), "https://api.euron.one/api/v1/euri/chat/completions")
            self.assertEqual(request.headers["authorization"], "Bearer test-key")
            payload = json.loads(request.content)
            self.assertEqual(payload["model"], "gemini-3.5-flash-lite")
            return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 0,
                "model": payload["model"], "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Grounded response [1]."}}]})
        client = EuriChat("test-key", "gemini-3.5-flash-lite", "https://api.euron.one/api/v1/euri")
        client.close()
        client.client = OpenAI(api_key="test-key", base_url="https://api.euron.one/api/v1/euri",
                               http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        try:
            self.assertEqual(client.complete([{"role": "user", "content": "Hi"}]), "Grounded response [1].")
        finally:
            client.close()


if __name__ == "__main__":
    unittest.main()
