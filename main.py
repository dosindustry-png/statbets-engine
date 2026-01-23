from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok", "message": "StatBets engine is running"}

@app.post("/daily-picks")
def daily_picks():
    return {
        "date": "test",
        "picks": []
    }
