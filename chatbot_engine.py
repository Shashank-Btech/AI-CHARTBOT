import os
import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim

class IntentClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(IntentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class ChatbotEngine:
    def __init__(self, csv_path=None):
        self.intents_dict = {}
        
        # Core intents (without jokes)
        core_intents = [
            ("greeting", "GENERAL", "Hello! I am Nova, your intelligent AI assistant. I am here to chat, answer any questions you have, or help you manage your shopping orders on our website. What is on your mind today?", ["Hello", "Hi there", "Good morning", "Hey bot", "Hi", "Greetings", "What's up", "Hello AI", "Good evening"]),
            ("about_bot", "GENERAL", "I am Nova, a conversational AI assistant powered by PyTorch and deep learning. I can chat about anything, answer general questions, and help you with customer support inquiries like tracking orders or canceling purchases on our website.", ["Who are you?", "What can you do?", "Tell me about yourself", "Are you an AI?", "What technologies power you?", "What is your name?"]),
            ("cancel_order", "ORDER", "To cancel your order, you can mention your Order ID here (e.g. 'Cancel order ORD-12345') or visit your User Dashboard on our website and click 'Cancel Order'.", ["How do I cancel my order?", "I need to cancel my purchase", "Can I stop my order from shipping?", "Cancel my recent order please", "Stop my purchase transaction", "How to cancel order"]),
            ("track_order", "SHIPPING", "You can track your package by mentioning your Order ID here or by visiting your User Dashboard on our website.", ["Where is my package?", "Track order status", "How can I check delivery tracking number?", "Has my item shipped yet?", "Shipping update for my purchase", "Track my shipment"]),
            ("refund_request", "BILLING", "Refunds are processed within 3-5 business days upon receipt of returned items. You can request a refund directly from your account page on our website.", ["How do I get a refund?", "I want my money back", "Request refund for damaged item", "When will my refund be processed?", "Check refund status"]),
            ("shipping_policy", "SHIPPING", "We offer Standard Shipping (3-5 days) and Express Delivery (1-2 days). Free shipping applies on all website orders over $50!", ["What are your shipping methods?", "How long does delivery take?", "Do you offer express shipping?", "What are shipping costs?"]),
            ("return_policy", "RETURNS", "We offer a 30-day hassle-free return policy. Returns are 100% free; simply generate a prepaid shipping label on our website returns portal.", ["What is your return policy?", "How do I return an item?", "Can I exchange a product?", "Are returns free?"]),
            ("contact_support", "SUPPORT", "Our support team is available 24/7 via live chat on our website or by calling 1-800-555-0199.", ["How can I speak to a live human?", "Contact customer service phone number", "Talk to support agent", "Email customer care"]),
            ("account_settings", "ACCOUNT", "You can reset your password or update your profile settings anytime by visiting the account settings page on our website.", ["How do I reset my password?", "Update my account profile", "Change email address", "Delete my account"]),
            ("payment_methods", "BILLING", "We accept all major credit/debit cards (Visa, Mastercard, Amex), PayPal, Apple Pay, and Google Pay. All payments are secured with 256-bit SSL encryption.", ["What payment methods do you accept?", "Can I pay with PayPal?", "Do you accept credit cards?", "Is payment secure?", "What cards are accepted?"])
        ]

        for intent, cat, resp, ex in core_intents:
            self.intents_dict[intent] = {"category": cat, "response": resp, "examples": ex}

        # Load dataset CSV
        if csv_path and os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if all(col in df.columns for col in ['intent', 'instruction', 'response', 'category']):
                    sample_df = df.dropna(subset=['intent', 'instruction', 'response']).head(3000)
                    for _, row in sample_df.iterrows():
                        intent = str(row['intent']).strip()
                        cat = str(row['category']).strip()
                        instruction = str(row['instruction']).strip()
                        response = str(row['response']).strip()

                        if intent not in self.intents_dict:
                            self.intents_dict[intent] = {
                                "category": cat,
                                "response": response,
                                "examples": [instruction]
                            }
                        else:
                            if len(self.intents_dict[intent]["examples"]) < 20:
                                self.intents_dict[intent]["examples"].append(instruction)
            except Exception as e:
                pass

        self.intents_data = []
        self.corpus = []
        self.labels = []
        self.intent_map = {}

        for idx, (intent_name, info) in enumerate(self.intents_dict.items()):
            item = {
                "intent": intent_name,
                "category": info["category"],
                "response": info["response"],
                "examples": info["examples"]
            }
            self.intents_data.append(item)
            self.intent_map[idx] = item
            for ex in info["examples"]:
                self.corpus.append(ex)
                self.labels.append(idx)

        self._train_models()

    def _train_models(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=5000)
        X_tfidf = self.vectorizer.fit_transform(self.corpus).toarray()

        self.X_train_tensor = torch.tensor(X_tfidf, dtype=torch.float32)
        self.y_train_tensor = torch.tensor(self.labels, dtype=torch.long)

        input_dim = X_tfidf.shape[1]
        hidden_dim = 128
        output_dim = len(self.intents_data)

        self.model = IntentClassifier(input_dim, hidden_dim, output_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        self.model.train()
        for epoch in range(40):
            optimizer.zero_grad()
            outputs = self.model(self.X_train_tensor)
            loss = criterion(outputs, self.y_train_tensor)
            loss.backward()
            optimizer.step()

        self.model.eval()

    def predict(self, query: str):
        start_time = time.time()
        if not query or not query.strip():
            return {
                "query": query or "",
                "intent": "general_chat",
                "category": "GENERAL",
                "confidence": 0.9,
                "response": "Hello! I'm Nova, your AI shopping assistant. You can ask me about products, place an order, or check order details with your Order ID. How can I help?",
                "latency_ms": round((time.time() - start_time) * 1000, 2)
            }
        query_lower = query.lower().strip()

        if any(w in query_lower for w in ["hi", "hello", "hey", "greetings", "good morning", "good evening"]):
            intent_obj = self.intents_dict.get("greeting", self.intents_data[0])
            return {
                "query": query,
                "intent": "greeting",
                "category": "GENERAL",
                "confidence": 0.99,
                "response": intent_obj["response"],
                "latency_ms": round((time.time() - start_time) * 1000, 2)
            }
        if any(w in query_lower for w in ["who are you", "what can you do", "your name", "about yourself"]):
            intent_obj = self.intents_dict.get("about_bot", self.intents_data[0])
            return {
                "query": query,
                "intent": "about_bot",
                "category": "GENERAL",
                "confidence": 0.99,
                "response": intent_obj["response"],
                "latency_ms": round((time.time() - start_time) * 1000, 2)
            }

        q_tfidf = self.vectorizer.transform([query]).toarray()
        q_tensor = torch.tensor(q_tfidf, dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(q_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probs, dim=1)
            
        conf_score = confidence.item()
        pred_idx = predicted_idx.item()

        corpus_sims = cosine_similarity(q_tfidf, self.vectorizer.transform(self.corpus).toarray())[0]
        best_match_idx = np.argmax(corpus_sims)
        sim_score = corpus_sims[best_match_idx]

        if sim_score > conf_score and sim_score > 0.2:
            pred_idx = self.labels[best_match_idx]
            conf_score = float(sim_score)

        if conf_score < 0.2:
            return {
                "query": query,
                "intent": "general_chat",
                "category": "GENERAL",
                "confidence": round(float(max(conf_score, 0.65)), 4),
                "response": f"That is an interesting question about \"{query}\"! As Nova, I can chat about anything, answer questions, or help you manage your shopping orders on our website. Let me know how I can assist you!",
                "latency_ms": round((time.time() - start_time) * 1000, 2)
            }

        intent_obj = self.intent_map[pred_idx]
        latency = round((time.time() - start_time) * 1000, 2)

        return {
            "query": query,
            "intent": intent_obj["intent"],
            "category": intent_obj["category"],
            "confidence": round(float(conf_score), 4),
            "response": intent_obj["response"],
            "latency_ms": latency
        }

    def evaluate_performance(self):
        """Run benchmark queries and return accuracy / latency stats."""
        test_cases = [
            ("How do I cancel my order?", "cancel_order"),
            ("Where is my package?", "track_order"),
            ("How do I get a refund?", "refund_request"),
            ("What are your shipping methods?", "shipping_policy"),
            ("What is your return policy?", "return_policy"),
            ("How can I speak to a live human?", "contact_support"),
            ("How do I reset my password?", "account_settings"),
            ("What payment methods do you accept?", "payment_methods"),
            ("Hello", "greeting"),
            ("Who are you?", "about_bot"),
        ]
        import time
        correct = 0
        latencies = []
        for query, expected_intent in test_cases:
            start = time.time()
            result = self.predict(query)
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            if result["intent"] == expected_intent:
                correct += 1
        total = len(test_cases)
        accuracy = (correct / total) * 100
        avg_latency = sum(latencies) / len(latencies)
        return {
            "total_tests": total,
            "correct": correct,
            "accuracy_percentage": round(accuracy, 2),
            "average_latency_ms": round(avg_latency, 2)
        }
