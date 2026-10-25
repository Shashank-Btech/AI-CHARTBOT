import sys
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_conversational_cancellation_flow():
    # 1. Create order
    order_payload = {
        "customerName": "Charlie Brown",
        "email": "charlie@example.com",
        "shippingAddress": "742 Peanuts Ln",
        "items": [{"productId": 4, "name": "Ultra-HD 4K Monitor", "price": 349.99, "quantity": 1}],
        "totalAmount": 349.99
    }
    res = client.post("/api/orders/create", json=order_payload)
    assert res.status_code == 200
    order_id = res.json()["orderId"]

    # 2. User mentions order ID in chat
    chat_res1 = client.post("/api/chat", json={"message": f"Where is order {order_id}?", "session_id": "test_user_1"})
    print("Turn 1 response:", chat_res1.json()["response"])
    assert order_id in chat_res1.json()["response"]
    assert "Would you like to cancel this order?" in chat_res1.json()["response"]

    # 3. User replies "cancel"
    chat_res2 = client.post("/api/chat", json={"message": "cancel", "session_id": "test_user_1"})
    print("Turn 2 response:", chat_res2.json()["response"])
    assert "successfully CANCELLED" in chat_res2.json()["response"]

    # 4. Verify dashboard list shows status as Cancelled
    list_res = client.get("/api/orders")
    orders = list_res.json()
    cancelled_found = False
    for o in orders:
        if o["orderId"] == order_id:
            if o["status"] == "Cancelled":
                cancelled_found = True
    assert cancelled_found

if __name__ == "__main__":
    try:
        test_conversational_cancellation_flow()
        print("Conversational order cancellation flow verified 100% successfully!")
    except Exception as e:
        print(f"Verification failed: {e}")
        sys.exit(1)
