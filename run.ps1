$ErrorActionPreference = "Stop"

if (-not (Test-Path ".env")) {
  Copy-Item ".env.example" ".env"
}

if (-not (Test-Path ".venv")) {
  python -m venv .venv
}

. .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

$env:PYTHONUTF8 = 1

python manage.py makemigrations
python manage.py migrate
python manage.py collectstatic --noinput

uvicorn finapps.asgi:app --host 127.0.0.1 --port 8000 --reload
