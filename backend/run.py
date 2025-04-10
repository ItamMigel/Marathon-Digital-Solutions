import uvicorn
from server import create_db

if __name__ == "__main__":
    create_db()
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True) 