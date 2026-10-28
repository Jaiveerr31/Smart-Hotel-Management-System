# Hotel Management System

Python Gradio application with SQLite for room management, bookings, billing, analytics and sentiment analysis.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

The SQLite database is created on first run. Do not commit database files containing guest details. The app has no access controls yet; do not publicly deploy with real guest data. A persistent database volume and access controls should be configured before production use.
