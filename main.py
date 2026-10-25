import os
import json
import random
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from chatbot_engine import ChatbotEngine

app = FastAPI(
    title="Nova AI Store & Assistant",
    description="Intelligent Conversational AI Chatbot with Stateful Order Management & Dashboard",
    version="3.3.0"
)

# Initialize Chatbot Engine
csv_file = os.path.join(os.path.dirname(__file__), "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")
chatbot = ChatbotEngine(csv_path=csv_file)

ORDERS_FILE = os.path.join(os.path.dirname(__file__), "orders.json")

def load_orders():
    if os.path.exists(ORDERS_FILE):
        try:
            with open(ORDERS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_orders(orders):
    try:
        with open(ORDERS_FILE, "w") as f:
            json.dump(orders, f, indent=4)
    except Exception as e:
        print(f"Error saving orders: {e}")

orders_db = load_orders()
session_order_context = {}

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default_user"

class ChatResponse(BaseModel):
    query: str
    intent: str
    category: str
    confidence: float
    response: str
    latency_ms: float

class OrderItem(BaseModel):
    productId: int
    name: str
    price: float
    quantity: int

class OrderCreateRequest(BaseModel):
    customerName: str
    email: str
    shippingAddress: str
    items: List[OrderItem]
    totalAmount: float

@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    global orders_db
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    
    try:
        result = chatbot.predict(request.message)
    except Exception as e:
        orders_db = load_orders()
        return ChatResponse(
            query=request.message,
            intent="general_chat",
            category="GENERAL",
            confidence=0.9,
            response=f"I understood what you said. As Nova, I can help you with your shopping orders, track shipments, or answer questions about our store. Try mentioning your Order ID or ask me about any product!",
            latency_ms=1.0
        )
    msg_upper = request.message.upper()
    session_id = request.session_id or "default_user"
    
    orders_db = load_orders()

    # Find if any order ID is mentioned in the message
    target_order_id = None
    for order_id in orders_db.keys():
        if order_id in msg_upper or order_id.replace("ORD-", "") in msg_upper:
            target_order_id = order_id
            break

    # If order ID found in message
    if target_order_id:
        session_order_context[session_id] = target_order_id
        order_info = orders_db[target_order_id]
        
        # If user explicitly asks to cancel in the same message
        if any(w in msg_upper for w in ["CANCEL", "STOP", "REVOKE"]):
            if order_info["status"] == "Cancelled":
                result["response"] = f"Order {target_order_id} is already cancelled."
            else:
                order_info["status"] = "Cancelled"
                orders_db[target_order_id] = order_info
                save_orders(orders_db)
                result["response"] = f"Success! Order {target_order_id} has been successfully CANCELLED as requested. This is now reflected in your Dashboard."
            result["intent"] = "cancel_order"
            if session_id in session_order_context:
                del session_order_context[session_id]
        else:
            # Show entire details and ask to cancel
            items_str = ", ".join([f"{i['quantity']}x {i['name']} (${i['price']:.2f} each)" for i in order_info["items"]])
            result["response"] = (
                f"Here are the entire details for order {target_order_id}:\n"
                f"- Customer Name: {order_info['customerName']}\n"
                f"- Email: {order_info['email']}\n"
                f"- Status: {order_info['status'].upper()}\n"
                f"- Items: {items_str}\n"
                f"- Total Amount: ${order_info['totalAmount']:.2f}\n"
                f"- Shipping Address: {order_info['shippingAddress']}\n\n"
                f"Would you like to cancel this order? (Type 'cancel' to confirm)"
            )
            result["intent"] = "order_lookup"

    # If no order ID in current message, but user is replying "cancel" to a previous order prompt in session
    elif any(w in msg_upper for w in ["CANCEL", "YES", "CONFIRM"]) and session_id in session_order_context:
        target_order_id = session_order_context[session_id]
        if target_order_id in orders_db:
            order_info = orders_db[target_order_id]
            if order_info["status"] == "Cancelled":
                result["response"] = f"Order {target_order_id} is already cancelled."
            else:
                order_info["status"] = "Cancelled"
                orders_db[target_order_id] = order_info
                save_orders(orders_db)
                result["response"] = f"Success! Order {target_order_id} has been successfully CANCELLED as requested. This is now reflected in your Dashboard."
            result["intent"] = "cancel_order"
            del session_order_context[session_id]

    return ChatResponse(**result)

@app.get("/api/intents")
def get_intents():
    return {
        "total_intents": len(chatbot.intents_data),
        "intents": chatbot.intents_data[:20]
    }

@app.get("/api/orders")
def list_orders():
    global orders_db
    orders_db = load_orders()
    return list(orders_db.values())

@app.post("/api/orders/create")
def create_order(order: OrderCreateRequest):
    global orders_db
    orders_db = load_orders()
    order_id = f"ORD-{random.randint(10000, 99999)}"
    order_data = order.dict()
    order_data["orderId"] = order_id
    order_data["status"] = "Processing"
    orders_db[order_id] = order_data
    save_orders(orders_db)
    return {"success": True, "orderId": order_id, "order": order_data}

@app.post("/api/orders/{order_id}/cancel")
def cancel_order(order_id: str):
    global orders_db
    orders_db = load_orders()
    order_id = order_id.upper()
    if order_id not in orders_db:
        raise HTTPException(status_code=404, detail="Order not found")
    orders_db[order_id]["status"] = "Cancelled"
    save_orders(orders_db)
    return {"success": True, "order": orders_db[order_id]}

@app.get("/api/orders/{order_id}")
def get_order(order_id: str):
    global orders_db
    orders_db = load_orders()
    order_id = order_id.upper()
    if order_id not in orders_db:
        raise HTTPException(status_code=404, detail="Order not found")
    return orders_db[order_id]

@app.get("/")
def read_root():
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Nova AI Chatbot is running."}

@app.get("/store")
def read_store():
    store_path = os.path.join(os.path.dirname(__file__), "static", "store.html")
    if os.path.exists(store_path):
        return FileResponse(store_path)
    return {"message": "Store page not found."}

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
