from fastapi import FastAPI, UploadFile, File
import os
from typing import List
from functions import get_results_func


app = FastAPI()

UPLOAD_FOLDER = "uploads"

# Create folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    saved_files = []

    for file in files:
        file_location = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())
        saved_files.append(file.filename)

    return {"uploaded_files": saved_files}


@app.get("/getResults")
def get_results():
    x = get_results_func()
    return x
