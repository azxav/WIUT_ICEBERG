# Demo API

The FastAPI service accepts MP4 uploads up to 200 MB and 120 seconds, resizes them to at most 1280×720, runs the mapped-junction event detector and causal risk estimator, then deletes temporary video files. Short demos allow 45 seconds for cold model startup in addition to the 3× clip-duration processing budget; the official submission harness still uses its unchanged 3× limit.

## Local development

From the repository root:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r demo_api/requirements.txt
python -m uvicorn demo_api.app:app --host 127.0.0.1 --port 8000
```

The Vite development server proxies `/api` requests to this local API.
