from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from agent import CollabFlowAgent

app = FastAPI(title="CollabFlow Agent API", version="1.0")

# Allow frontend access (CORS)
app.add_middleware(
    CORSMiddleware,
    # change this to your frontend domain in production
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Initialize Agent ---------- #
agent = CollabFlowAgent()
agent.initialize()


# ---------- Request & Response Models ---------- #
class MessageRequest(BaseModel):
    session_id: str
    message: str


class MessageResponse(BaseModel):
    session_id: str
    response: str


# ---------- Routes ---------- #
@app.get("/")
async def root():
    return {"status": "ok", "message": "CollabFlow Agent API is running 🚀"}


@app.post("/chat", response_model=MessageResponse)
async def chat_endpoint(request: MessageRequest):
    """
    Send a user message to the agent.
    Each session_id maintains its own conversation context.
    """
    try:
        response_text = agent.send_message(
            request.message, session_id=request.session_id)
        return MessageResponse(session_id=request.session_id, response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Run Server ---------- #
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
