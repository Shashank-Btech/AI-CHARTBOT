import unittest
from chatbot_engine import ChatbotEngine

class TestChatbotEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = ChatbotEngine()

    def test_intents_count(self):
        self.assertGreaterEqual(len(self.engine.intents_data), 10, "Must support at least 10 intents")

    def test_prediction_output(self):
        res = self.engine.predict("How do I cancel my order?")
        self.assertIn("intent", res)
        self.assertIn("response", res)
        self.assertIn("confidence", res)
        self.assertEqual(res["intent"], "cancel_order")
        self.assertGreater(res["confidence"], 0.0)

    def test_evaluation_accuracy(self):
        eval_res = self.engine.evaluate_performance()
        self.assertGreaterEqual(eval_res["total_tests"], 10)
        self.assertGreaterEqual(eval_res["accuracy_percentage"], 80.0, "Accuracy should be at least 80%")
        print(f"\n[Test Result] Accuracy: {eval_res['accuracy_percentage']}%, Avg Latency: {eval_res['average_latency_ms']}ms")

if __name__ == "__main__":
    unittest.main()
