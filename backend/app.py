import uvicorn

# Entrypoint convenience so `python app.py` works the same as `uvicorn main:app`.
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
