# Clustering Capacitado API (FastAPI) – Heroku

## Rodar localmente
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
set API_TOKEN=meu_token_supersecreto  # Windows
# export API_TOKEN=meu_token_supersecreto  # macOS/Linux
uvicorn app_fastapi:app --reload --host 0.0.0.0 --port 8000
