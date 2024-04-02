from main import util_bot
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional



app = FastAPI()

class QueryModel(BaseModel):
    question: str

@app.get("/")
async def read_root():
    return {"message": "Util Bot API!"}

@app.post("/query/")
async def query_bot(query: QueryModel):
    try:
        response = util_bot(question=query.question)
        if response is None:
            return {"answer": "Unable to generate an answer."}
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

