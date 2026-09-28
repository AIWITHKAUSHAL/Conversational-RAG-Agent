import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


class AppTests(unittest.TestCase):
    def test_sample_chat_followup_and_reset(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {
            "RAG_DB_PATH": str(Path(folder) / "ui.db"), "EURI_API_KEY": "test-key"
        }):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
            self.assertFalse(app.exception)
            self.assertTrue(app.chat_input[0].disabled)
            next(b for b in app.button if b.label == "Load sample: Agentic AI").click().run()
            self.assertFalse(app.chat_input[0].disabled)
            with patch("rag.EuriChat.complete", side_effect=[
                "Agentic AI uses tools [1].", "Agentic AI benefits", "It automates tasks [1]."
            ]):
                app.chat_input[0].set_value("What is Agentic AI?").run()
                app.chat_input[0].set_value("What are its benefits?").run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.chat_message), 4)
            next(b for b in app.button if b.label == "Clear conversation memory").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.chat_message), 0)
            self.assertFalse(app.chat_input[0].disabled)


if __name__ == "__main__":
    unittest.main()
