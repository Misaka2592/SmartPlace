# SmartPlace FastAPI Backend

## Start

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

## Notes

- Existing `app.py` is preserved as the Gradio backup version.
- Uploaded files, sessions, and run metadata are stored under `backend/storage/`.
- Generated reports, composites, masks, and explanation artifacts still reuse the existing project output folders.
