import subprocess
import sys
import os
import time
import requests

# Kill any existing python processes running run_server.py or uvicorn on port 8000
os.system("taskkill /f /im python.exe /fi \"WINDOWTITLE eq *uvicorn*\" >nul 2>&1")

# Launch uvicorn in a new console window so it runs persistently in the background
cmd = f'start "Uvicorn Chatbot Server" cmd /k "{sys.executable} -m uvicorn main:app --host 127.0.0.1 --port 8000"'
subprocess.Popen(cmd, shell=True)

print("Waiting for server to initialize (loading ML model)...")
time.sleep(8)

for attempt in range(1, 6):
    try:
        res = requests.get("http://127.0.0.1:8000/api/intents", timeout=5)
        if res.status_code == 200:
            print(f"SUCCESS: Server is running persistently at http://127.0.0.1:8000 (attempt {attempt})")
            break
        else:
            print(f"Attempt {attempt}: Server responded with status code {res.status_code}")
    except Exception as e:
        print(f"Attempt {attempt}: Not ready yet, retrying in 2s... ({e})")
        time.sleep(2)
else:
    print("WARNING: Server did not respond after 5 attempts. Check the CMD window for errors.")
