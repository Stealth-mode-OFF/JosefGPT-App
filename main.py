from fastapi import FastAPI, Query
from agent_retriever_http_fix import ask_agent

app = FastAPI()

@app.get("/ask")
def ask(q: str = Query(..., description="Question for the agent")):
    try:
        answer = ask_agent(q)
        return {"query": q, "answer": answer}
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health():
    return {"status": "ok"}
