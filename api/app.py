from fastapi import FastAPI, Depends, HTTPException, status, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict

# Initialize Django so ORM can be used inside FastAPI
import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'finapps.settings')
django.setup()

from django.contrib.auth.models import User  # noqa: E402
from apps.accounts.models import APIToken, Profile  # noqa: E402
from apps.banking.models import BankConnection  # noqa: E402
from apps.insights.ml_engine import compute_insights_from_url  # noqa: E402

fastapi_app = FastAPI(title="FinApps API", version="0.1.0")
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InsightsResponse(BaseModel):
    data: Dict[str, Any]


def get_current_user(authorization: str | None = Header(default=None)) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    token_value = authorization.split(" ", 1)[1].strip()
    try:
        token_obj = APIToken.objects.select_related('user').get(token=token_value)
    except APIToken.DoesNotExist:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return token_obj.user


@fastapi_app.get("/health")
def health():
    return {"ok": True}


@fastapi_app.get("/insights", response_model=InsightsResponse)
def get_insights(user: User = Depends(get_current_user)):
    profile = Profile.objects.filter(user=user).first()
    if not profile:
        raise HTTPException(status_code=400, detail="Profile not found")
    bank = BankConnection.objects.filter(user=user).first()
    if not bank or not bank.csv_url:
        raise HTTPException(status_code=400, detail="Bank connection/CSV not found")
    try:
        insights = compute_insights_from_url(bank.csv_url, float(profile.monthly_income))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to compute insights: {e}")
    return {"data": insights}
