from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os
from datetime import datetime

app = FastAPI()

# ✅ CORS so your laptop HTML can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # for testing; later you can lock this down
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SAVE_DIR = "uploads"
os.makedirs(SAVE_DIR, exist_ok=True)

@app.get("/health")
def health():
    return {"ok": True, "message": "backend is reachable"}

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    seconds: str = Form(None),
    user_id: str = Form(None),
):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_user = (user_id or "user").replace("/", "_")
    filename = f"{safe_user}_{file.filename}"
    path = os.path.join(SAVE_DIR, filename)

    # ✅ Save file
    data = await file.read()
    with open(path, "wb") as f:
        f.write(data)

    size_kb = len(data) / 1024.0

    output="model is runnning and your model is good"
    return {
        "ok": True,
        "message":output,
        "saved_path": path,
    }
