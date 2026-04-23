# HP-API

---

## Overview

This API allows you to:

1. Upload input files (e.g. CSV)
2. Run a pipeline that extracts triples and explanations
3. Retrieve the results as JSON

---

## Run the API

### Without Docker

```bash
python -m uvicorn main:app --reload
```

---

### With Docker

```bash
docker build -t hpapi .
docker run -p 8000:8000 hpapi
```

---

### Run with GPU (optional)

```bash
docker run --gpus all -p 8000:8000 hpapi
```

---

## Base URL

```text
http://127.0.0.1:8000
```

---

## Endpoints

### 1. Upload files

**POST** `/upload`

Uploads one or more files and stores them in the `uploads/` folder.

#### Request

Multipart form-data with field name **files**

#### Example

```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -F "files=@files/case.csv" \
  -F "files=@files/example.csv"
```

---

### 2. Get results

**GET** `/getResults`

Runs the pipeline and returns triples with explanations as JSON.

#### Example

```bash
curl "http://127.0.0.1:8000/getResults"
```

---

## Project Structure

| File/Folder  | Description                             |
| ------------ | --------------------------------------- |
| main.py      | FastAPI app and routes                  |
| functions.py | Runs the pipeline                       |
| code/        | Triple extraction and explanation logic |
| files/       | Input & output files                    |
| uploads/     | Uploaded files                          |

---
