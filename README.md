# Intelligent Conversational AI Chatbot & E-Commerce Store (Nova AI)

An enterprise-grade, intelligent NLP chatbot application integrated with a full-featured e-commerce store and persistent user dashboard. Powered by **Python**, **PyTorch**, **Transformers**, **LangChain principles**, and **FastAPI**.

---

## 🚀 Key Features

1. **Intelligent NLP & Intent Recognition**:
   - Powered by a custom **PyTorch neural network intent classifier** and **TF-IDF / Cosine Similarity** vector engine trained on customer support datasets.
   - Supports natural language understanding for customer inquiries, greetings, product questions, order tracking, and cancellations.

2. **E-Commerce Shopping Website**:
   - Modern storefront featuring high-resolution product photography (Unsplash), rich item descriptions, shopping cart, and instant checkout.
   - Generates unique Order IDs (e.g., `ORD-84921`) upon purchase.

3. **Persistent User Orders Dashboard**:
   - Stores all placed orders persistently on disk (`orders.json`) and browser `localStorage`.
   - Real-time tracking of order status (`Processing` or `Cancelled`), itemized lists, totals, and shipping addresses.

4. **Stateful Conversation Flow & Order Management**:
   - **Entire Details Retrieval**: Mentioning an Order ID in chat prompts Nova AI to fetch and display the complete order breakdown.
   - **Interactive Cancellation**: Nova AI asks *"Would you like to cancel this order? (Type 'cancel' to confirm)"*. Typing `cancel` immediately updates the order status to **Cancelled** across both backend storage and the user dashboard.

---

## 🛠️ Technology Stack

- **Python 3.11+**
- **PyTorch** (Neural Network Intent Classifier)
- **Transformers / Hugging Face & LangChain** (NLP & Retrieval Architecture)
- **FastAPI & Uvicorn** (REST API Backend & Static Web Server)
- **scikit-learn** (TF-IDF Vectorizer & Cosine Similarity)
- **HTML5 / CSS3 / JavaScript (Modern UI with Dark Theme)**

---

## 🔄 Conversation Flow Documentation

```
[User Message] 
       │
       ▼
[FastAPI /api/chat Endpoint]
       │
       ├──► 1. Intent Classification (PyTorch / TF-IDF Model)
       │         └── Matches customer support intent or general chat.
       │
       ├──► 2. Order ID Detection (Regex / Pattern Matching)
       │         ├── If Order ID found:
       │         │     ├── Fetches entire details from persistent storage (orders.json).
       │         │     ├── Displays complete breakdown (Customer, Items, Total, Address).
       │         │     └── Asks: "Would you like to cancel this order? (Type 'cancel' to confirm)"
       │         │
       │         └── If user replies "cancel" (or says "Cancel order ORD-XXXXX"):
       │               ├── Updates order status to "Cancelled" in backend storage.
       │               ├── Confirms cancellation in chat.
       │               └── Instantly reflects "Cancelled" status in User Dashboard table.
       │
       └──► 3. Response Generation & Latency Metrics returned to UI.
```

---

## 🏃‍♂️ How to Run the Application

1. **Install Dependencies**:
   ```bash
   pip install fastapi uvicorn torch transformers sentence-transformers scikit-learn pandas requests
   ```

2. **Start the FastAPI Server**:
   ```bash
   python main.py
   ```
   *(Or run `python launch_server.py` to start the server persistently).*

3. **Access the Application**:
   - **E-Commerce Store & Dashboard**: Open [http://127.0.0.1:8000/store](http://127.0.0.1:8000/store) in your browser.
   - **Nova AI Chat Assistant UI**: Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser.
   - **Interactive API Documentation (Swagger)**: Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
