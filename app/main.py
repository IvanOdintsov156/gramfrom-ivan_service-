import uvicorn
from fastapi import FastAPI
from app.routers import similarity, gramform

app = FastAPI()

app.include_router(similarity.router)
app.include_router(gramform.router)

#if __name__ == "__main__":
#    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)