---
title: WIUT Traffic Event Demo API
sdk: docker
app_port: 7860
---

# Demo API

Build from the repository root with `demo_api/Dockerfile`. The API accepts MP4 uploads up to 200 MB and 120 seconds, resizes them to at most 1280×720, runs the mapped-junction event detector and causal risk estimator, then deletes temporary video files. Short demos allow 45 seconds for cold model startup in addition to the 3× clip-duration processing budget; the official submission harness still uses its unchanged 3× limit.

Set `DEMO_CORS_ORIGINS` to the exact Vercel site origin before deployment. Keep the Space private until the team approves publication and checks its model/data licences.

Local run, from the repository root:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r demo_api/requirements.txt
python -m uvicorn demo_api.app:app --host 127.0.0.1 --port 8000
```
