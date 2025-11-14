from __future__ import annotations
import os
import math
from typing import Any, Dict, List, Tuple
import time
import httpx
from django.conf import settings
try:
    from duckduckgo_search import DDGS
except Exception:  # pragma: no cover
    DDGS = None  # type: ignore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover
    BeautifulSoup = None  # type: ignore
try:
    import google.generativeai as genai
except Exception:  # pragma: no cover
    genai = None  # type: ignore
import re
import json
try:
    from langgraph.graph import StateGraph, END
except Exception:  # pragma: no cover
    StateGraph, END = None, None  # type: ignore


FASTAPI_BASE = os.getenv("FINAPPS_API_BASE", "http://127.0.0.1:8000")
DISABLE_HEALTH_SCRAPE = os.getenv("DISABLE_HEALTH_SCRAPE", "0") == "1"
OR_CONNECT_TIMEOUT = float(os.getenv("OPENROUTER_CONNECT_TIMEOUT", "2.0"))
OR_READ_TIMEOUT = float(os.getenv("OPENROUTER_READ_TIMEOUT", "4.0"))
OR_ATTEMPTS_PER_MODEL = int(os.getenv("OPENROUTER_ATTEMPTS_PER_MODEL", "1"))
OR_MAX_MODELS = int(os.getenv("OPENROUTER_MAX_MODELS", "1"))


def fetch_insights_via_api(token: str) -> Dict[str, Any]:
    url = f"{FASTAPI_BASE}/api/insights"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        with httpx.Client(timeout=6.0) as client:
            r = client.get(url, headers=headers)
            r.raise_for_status()
            payload = r.json()
            return payload.get("data", {})
    except Exception:
        return {}


def ddg_health_insurance_results(limit: int = 5) -> List[Dict[str, str]]:
    q = "best health insurance plans India 2025 site:*.in"
    results: List[Dict[str, str]] = []
    try:
        if DDGS is None:
            return []
        with DDGS() as ddgs:
            for r in ddgs.text(q, max_results=min(limit, 3)):
                results.append({
                    "title": r.get("title", ""),
                    "href": r.get("href", ""),
                    "body": r.get("body", ""),
                })
    except Exception:
        return []
    return results


PRICE_RE = re.compile(r"(?:₹|Rs\.?|INR)\s*([\d,]+)")


def _extract_inr(text: str) -> float | None:
    if not text:
        return None
    m = PRICE_RE.search(text.replace("\xa0", " "))
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except Exception:
        return None


def _latest_numeric(d: Dict[str, Any]) -> float:
    if not isinstance(d, dict) or not d:
        return 0.0
    try:
        k = sorted(d.keys())[-1]
        return float(d.get(k) or 0)
    except Exception:
        try:
            return float(list(d.values())[-1])
        except Exception:
            return 0.0


def _education_cards_from_insights(insights: Dict[str, Any]) -> List[Dict[str, Any]]:
    income = 0.0
    try:
        income = float(insights.get('income_per_month') or 0)
    except Exception:
        income = 0.0
    saved = _latest_numeric(insights.get('monthly_savings') or insights.get('savings_by_month') or {})
    savings_rate = (saved / income) if income else 0.0
    if savings_rate < 0.08:
        return [
            {"title": "SIP Basics", "subtitle": "Start small, stay consistent", "highlights": [
                "Pick a SIP you can sustain even in tight months",
                "Begin with ₹1k–2k and step-up yearly",
                "Use auto-debit to build discipline"
            ]},
            {"title": "Budgeting 50-30-20", "subtitle": "Spend-Save balance", "highlights": [
                "50% needs, 30% wants, 20% investing",
                "Trim 1–2 variable expenses to free SIP cash",
                "Review monthly and adjust"
            ]},
        ]
    elif savings_rate < 0.18:
        return [
            {"title": "SIP Ladder", "subtitle": "Split across goals", "highlights": [
                "Liquid/Debt for near-term needs",
                "Index for core long-term growth",
                "ELSS only if tax-saving under 80C needed"
            ]}
        ]
    else:
        return [
            {"title": "Index vs ELSS", "subtitle": "Core vs tax-saving", "highlights": [
                "Index: low-cost, diversified long-term core",
                "ELSS: tax-saving (80C) but 3y lock-in",
                "Combine if you have 80C room"
            ]}
        ]


def _advisor_plan_from_insights(insights: Dict[str, Any]) -> Dict[str, Any]:
    def round_step(x: float, step: int = 500) -> int:
        if x <= 0:
            return step
        return max(step, int(round(x / step) * step))

    try:
        income = float(insights.get('income_per_month') or 0)
    except Exception:
        income = 0.0
    saved = _latest_numeric(insights.get('monthly_savings') or insights.get('savings_by_month') or {})
    savings_rate = (saved / income) if income else (0.0 if saved == 0 else 0.15)
    # Heuristic risk
    risk_score = None
    try:
        risk_score = float(insights.get('risk_score'))
    except Exception:
        risk_score = None
    if risk_score is None:
        risk_score = 35 if savings_rate < 0.08 else (52 if savings_rate < 0.18 else 65)

    # Recommend category
    if risk_score < 40:
        category = 'Liquid Fund'
        expected = 6
        lock_in = 0
        duration = 3
    elif risk_score < 60:
        category = 'Index Fund'
        expected = 11
        lock_in = 0
        duration = 10
    else:
        category = 'Index Fund'
        expected = 12
        lock_in = 0
        duration = 10

    # Amount from savings/income
    base_amt = saved * 0.6 if saved > 0 else income * 0.1
    amount_inr = round_step(base_amt if base_amt > 0 else 2000)

    # Pick a matching product
    products = get_invest_products()
    chosen = None
    for p in products:
        if str(p.get('category')) == category and p.get('top'):
            chosen = p; break
    if not chosen:
        for p in products:
            if str(p.get('category')) == category:
                chosen = p; break
    company = (chosen or {}).get('company') or 'HDFC Mutual Fund'
    product = (chosen or {}).get('product') or ('HDFC Index Fund - Nifty 50 Plan' if category=='Index Fund' else 'HDFC Liquid Fund')

    return {
        "company": company,
        "product": product,
        "investment_type": category,
        "amount_inr": int(amount_inr),
        "frequency": "month",
        "risk_score": int(risk_score),
        "why_this": (
            "Low-cost, diversified core holding" if category=='Index Fund' else
            "Near-term safety and liquidity for parking cash"
        ),
        "expected_return_pct": expected,
        "lock_in_years": lock_in,
        "duration_years": duration,
    }


def education_cards_from_insights(insights: Dict[str, Any]) -> List[Dict[str, Any]]:
    return _education_cards_from_insights(insights)


def advisor_plan_from_insights(insights: Dict[str, Any]) -> Dict[str, Any]:
    return _advisor_plan_from_insights(insights)

KNOWN_BRAND_URLS = {
    "acko": "https://www.acko.com/health-insurance/",
    "niva bupa": "https://www.nivabupa.com/health-insurance.html",
    "care": "https://www.careinsurance.com/health-insurance/",
    "star health": "https://www.starhealth.in/",
    "icici lombard": "https://www.icicilombard.com/health-insurance",
    "hdfc ergo": "https://www.hdfcergo.com/health-insurance",
    "tata aig": "https://www.tataaig.com/health-insurance",
    "bajaj allianz": "https://www.bajajallianz.com/health-insurance.html",
}


def _scrape_plan_from_url(url: str) -> List[Dict[str, Any]]:
    plans: List[Dict[str, Any]] = []
    try:
        if BeautifulSoup is None:
            return plans
        with httpx.Client(timeout=10.0, headers={"User-Agent": "Mozilla/5.0"}) as client:
            r = client.get(url)
            if r.status_code != 200 or 'text/html' not in r.headers.get('Content-Type', ''):
                return plans
            soup = BeautifulSoup(r.text, 'html.parser')
            title = (soup.title.text if soup.title else "").strip()
            og_title = soup.find('meta', attrs={'property': 'og:title'})
            if og_title:
                title = og_title.get('content', title).strip()
            price_nodes = soup.find_all(text=PRICE_RE)
            price = None
            for node in price_nodes[:3]:
                price = _extract_inr(node)
                if price:
                    break
            brand = title.split('|')[0].split('–')[0].split('-')[0].strip() or url.split('/')[2]
            duration = None
            dur_node = soup.find(text=re.compile(r"\b(year|month|policy term)\b", re.I))
            if dur_node:
                duration = dur_node.strip()[:60]
            plans.append({
                'brand': brand,
                'premium_inr': price,
                'duration': duration or '1 year',
                'apply_link': url,
            })
    except Exception:
        return plans
    return plans


def discover_health_plans(insights: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    # Queries across known brands/aggregators
    queries = [
        'site:acko.com health insurance plans',
        'site:nivabupa.com health insurance plans',
        'site:careinsurance.com health insurance plans',
        'site:starhealth.in health insurance',
        'site:icicilombard.com health insurance',
        'site:hdfcergo.com health insurance',
        'site:tataaig.com health insurance',
        'site:bajajallianz.com health insurance',
    ]
    found: List[Dict[str, Any]] = []
    try:
        if not DISABLE_HEALTH_SCRAPE and DDGS is not None:
            with DDGS() as ddgs:
                for q in queries:
                    for r in ddgs.text(q, max_results=1):
                        url = r.get('href', '')
                        if not url:
                            continue
                        scraped = _scrape_plan_from_url(url)
                        if scraped:
                            found.extend(scraped)
        else:
            found = []
    except Exception:
        found = []

    # Fallback seeded brands if scraping sparse
    if len(found) < 6:
        for brand, link in KNOWN_BRAND_URLS.items():
            est = 7000.0  # base estimate
            if insights:
                avg_spend = sum(insights.get('monthly_spendings', {}).values()) / (len(insights.get('monthly_spendings', {})) or 1)
                est = max(4000.0, min(18000.0, 0.25 * (insights.get('income_per_month', 40000) or 40000) - 0.05 * avg_spend))
            found.append({'brand': brand.title(), 'premium_inr': round(est, 0), 'duration': '1 year', 'apply_link': link})

    # De-duplicate by brand name (case-insensitive)
    dedup: Dict[str, Dict[str, Any]] = {}
    for p in found:
        key = str(p.get('brand', '')).strip().lower()
        if key and key not in dedup:
            dedup[key] = p

    base = list(dedup.values())[:12]

    # Segment into groups with slight premium multipliers
    def segment(mult: float) -> List[Dict[str, Any]]:
        seg = []
        for p in base:
            price = p.get('premium_inr')
            if price:
                price = round(float(price) * mult, 0)
            seg.append({**p, 'premium_inr': price})
        return seg

    plans = {
        'self': segment(1.0),
        'spouse': segment(1.05),
        'child': segment(0.9),
        'parents': segment(1.4),
    }
    return plans


def build_gemini() -> ChatGoogleGenerativeAI:
    key = settings.GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Gemini API key missing. Set GEMINI_API_KEY in .env")
    # Ensure downstream libs also see the key even if constructor signature differs
    os.environ["GOOGLE_API_KEY"] = key
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=key,
            temperature=0.2,
        )
    except TypeError:
        # Older/newer versions might only read from env
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.2,
        )


def _gemini_direct_infer(prompt: str, key: str) -> str:
    if genai is None:
        raise RuntimeError("google-generativeai package not available")
    genai.configure(api_key=key)
    last_err = None
    for model in ("gemini-1.5-flash", "models/gemini-1.5-flash", "gemini-1.5-pro"):
        try:
            gm = genai.GenerativeModel(model)
            r = gm.generate_content(prompt)
            if hasattr(r, 'text') and r.text:
                return r.text
            if hasattr(r, 'candidates') and r.candidates:
                return str(r.candidates[0])
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Gemini direct failed: {last_err}")


def build_openrouter(model: str | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.OPENROUTER_API_KEY,
        base_url=settings.OPENROUTER_BASE_URL,
        model=(model or getattr(settings, 'OPENROUTER_MODEL', None) or "deepseek/deepseek-chat-v3.1:free"),
        temperature=0.2,
    )


OPENROUTER_FALLBACK_MODELS: List[str] = [
    getattr(settings, 'OPENROUTER_MODEL', None) or "deepseek/deepseek-chat-v3.1:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "mistralai/mixtral-8x7b-instruct:free",
    "qwen/qwen-2.5-7b-instruct:free",
    "google/gemma-2-9b-it:free",
]


def openrouter_invoke(prompt: str) -> str:
    """Call OpenRouter with HTTP first (headers set), then LangChain fallback across models."""
    # Primary: direct HTTP call (better header control)
    base = getattr(settings, 'OPENROUTER_BASE_URL', None) or os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
    key = getattr(settings, 'OPENROUTER_API_KEY', None) or os.getenv('OPENROUTER_API_KEY')
    referer = os.getenv('OPENROUTER_REFERER', 'http://127.0.0.1:8000')
    title = os.getenv('OPENROUTER_TITLE', 'FinApps')
    if key:
        models = OPENROUTER_FALLBACK_MODELS[:max(1, OR_MAX_MODELS)]
        for m in models:
            # retry per-model with backoff
            for attempt in range(max(1, OR_ATTEMPTS_PER_MODEL)):
                try:
                    with httpx.Client(timeout=httpx.Timeout(connect=OR_CONNECT_TIMEOUT, read=OR_READ_TIMEOUT, write=OR_READ_TIMEOUT, pool=10.0)) as client:
                        r = client.post(
                            f"{base}/chat/completions",
                            headers={
                                "Authorization": f"Bearer {key}",
                                "HTTP-Referer": referer,
                                "X-Title": title,
                                "Content-Type": "application/json",
                            },
                            json={
                                "model": m,
                                "messages": [{"role": "user", "content": prompt}],
                                "temperature": 0.2,
                            },
                        )
                        if r.status_code in (429, 502, 503, 504):
                            # backoff then retry same model
                            time.sleep(1.5 * (attempt + 1))
                            continue
                        r.raise_for_status()
                        data = r.json()
                        choices = (data or {}).get("choices") or []
                        if choices:
                            msg = choices[0].get("message", {})
                            content = msg.get("content")
                            if content:
                                return content
                        # if no choices, break to next model
                        break
                except Exception:
                    # connection/read errors -> backoff and retry
                    time.sleep(1.2 * (attempt + 1))
                    continue
    # Secondary: LangChain client across models
    for m in OPENROUTER_FALLBACK_MODELS:
        try:
            llm = build_openrouter(model=m)
            resp = llm.invoke(prompt)
            return resp.content if hasattr(resp, "content") else str(resp)
        except Exception:
            continue
    return "Investment assistant is temporarily unavailable. Please try again shortly."


def health_agent_reply(insights: Dict[str, Any], user_prompt: str) -> str:
    system_text = (
        "You are a health insurance advisor for India. Only answer questions about health insurance, benefits, coverage, premiums, policy terms, claims process, exclusions, and tax benefits under 80D. "
        "Use the provided financial insights if relevant to personalize suggestions. "
        "Be concise and actionable. Always note that prices vary by age, city, riders, and underwriting. "
        "Language policy: Reply in Hindi if the user's query is in Hindi; otherwise reply in English."
    )
    msg = (
        f"Insights: {insights}\n\n"
        f"User question: {user_prompt}\n"
        "Return a short helpful answer with clear bullets. At the end, also output a JSON in a fenced block ```json ... ``` with key 'cards' as an array. "
        "Each card has: title (string), subtitle (string, optional), highlights (array of short strings), cta_label (string, optional), cta_url (string, optional)."
    )
    # Primary: direct google-generativeai (more reliable across versions)
    try:
        key = settings.GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY")
        os.environ["GOOGLE_API_KEY"] = key or ""
        direct = _gemini_direct_infer("System: " + system_text + "\n\n" + msg, key or "")
        return direct
    except Exception:
        # Secondary: LangChain Gemini
        try:
            llm = build_gemini()
            prompt = ("System: " + system_text + "\n\n" + msg)
            r = llm.invoke(prompt)
            return r.content if hasattr(r, "content") else str(r)
        except Exception:
            # Fallback: OpenRouter (generic LLM)
            try:
                prompt3 = ("System: " + system_text + "\n\n" + msg)
                resp3 = openrouter_invoke(prompt3)
                return resp3
            except Exception:
                return "We’re temporarily unable to reach the health chatbot. Please try again in a minute."


NBFC_LIST = [
    "Bajaj Finance", "Tata Capital", "Mahindra Finance", "HDB Financial Services",
    "Aditya Birla Finance", "Muthoot Finance", "Manappuram Finance", "L&T Finance",
    "Cholamandalam Investment and Finance", "Fullerton India", "Shriram Finance",
]


def loan_agent_reply(insights: Dict[str, Any], user_prompt: str) -> str:
    prompt = (
        "System: You are a loan recommendation assistant for India. Given the user's financial insights and NBFC list, suggest suitable loan products, APR range, and eligibility tips.\n\n"
        f"NBFCs: {NBFC_LIST}\n"
        f"Insights: {insights}\n"
        f"User question: {user_prompt}\n"
        "Return concise bullet points."
    )
    try:
        return openrouter_invoke(prompt)
    except Exception as e:
        return "Loan assistant is temporarily unavailable. Please try again shortly."


FUND_CATEGORIES = ["Liquid Funds", "Large Cap Equity", "Mid Cap Equity", "Small Cap Equity", "ELSS", "Index Funds", "Hybrid Funds", "Debt Funds"]


def sip_projection(monthly_sip: float, annual_rate: float, years: int) -> Tuple[List[int], List[float]]:
    r = annual_rate / 12.0
    n = years * 12
    values = []
    years_x = list(range(1, years + 1))
    for y in years_x:
        m = y * 12
        fv = monthly_sip * (((1 + r) ** m - 1) / r) * (1 + r)
        values.append(round(fv, 2))
    return years_x, values


def invest_agent_reply(insights: Dict[str, Any], user_prompt: str) -> str:
    prompt = (
        "System: You are an investment planning assistant for India. Be concise, friendly, and educational. Use a bilingual tone (English + simple Hindi). Add relevant emojis.\n"
        "Focus on mutual funds, liquid funds, debt funds, index funds, gold ETFs, and ELSS only. Do not provide stock tips.\n"
        "When asked for a plan, include amounts and frequency, explain risk briefly, and provide expected return and lock-in (if any).\n"
        "At the end of your message, output a JSON object in a fenced block ```json ... ``` with keys: company, product, investment_type, amount_inr, frequency, risk_score, why_this, expected_return_pct, lock_in_years, duration_years. Use snake_case keys only.\n\n"
        f"Categories: {FUND_CATEGORIES}\n"
        f"Insights: {insights}\n"
        f"User: {user_prompt}\n"
        "Respond bilingually with clear bullets and emojis."
    )
    # Try OpenRouter first (primary for investment)
    try:
        resp = openrouter_invoke(prompt)
        if resp and not resp.strip().lower().startswith("investment assistant is temporarily unavailable"):
            return resp
    except Exception:
        pass

    # Fallback to Gemini to keep the assistant working
    try:
        key = settings.GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY")
        os.environ["GOOGLE_API_KEY"] = key or ""
        direct = _gemini_direct_infer(prompt, key or "")
        return direct
    except Exception:
        try:
            llm2 = build_gemini()
            r2 = llm2.invoke(prompt)
            return r2.content if hasattr(r2, "content") else str(r2)
        except Exception:
            return "Investment assistant is temporarily unavailable. Please try again shortly."


# ------- Investment Multi-Agent (LangGraph) -------

def _extract_fenced_json(text: str) -> Dict[str, Any] | None:
    if not text:
        return None
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def _education_agent_prompt(insights: Dict[str, Any], q: str) -> str:
    return (
        "System: You are an Education Agent that explains investing concepts in simple terms with examples for Indian investors. Use clear bullets and tiny ASCII illustrations when helpful.\n"
        "Also include a fenced JSON block ```json {\"cards\": [{title, subtitle?, highlights[] }]} ``` for UI cards.\n\n"
        f"Insights: {insights}\n"
        f"User: {q}"
    )


def _analyzer_agent_prompt(insights: Dict[str, Any], q: str) -> str:
    return (
        "System: You are a Data Analyzer Agent. Analyze the user's income, spend, savings, and risk signals from the insights."
        " Output a friendly summary like 'You saved ₹X this month; expense ratio improved by Y%'."
        " At the end include a fenced JSON block ```json {\"metrics\": {\"saved_inr\": number, \"expense_ratio_change_pct\": number}} ``` .\n\n"
        f"Insights: {insights}\n"
        f"Focus: {q or 'overall'}\n"
    )


def _advisor_agent_prompt(insights: Dict[str, Any], q: str) -> str:
    return (
        "System: You are an Investment Advisor Agent for India. Suggest a plan aligned to user's goal and risk."
        " Focus on mutual funds (index, ELSS, liquid, debt, gold ETF). No stock tips."
        " Include amounts, frequency, expected return, and lock-in if any."
        " End with a fenced JSON block ```json {company, product, investment_type, amount_inr, frequency, risk_score, why_this, expected_return_pct, lock_in_years, duration_years} ``` .\n\n"
        f"Insights: {insights}\nUser: {q}"
    )


def _goal_tracker_agent_prompt(insights: Dict[str, Any], q: str) -> str:
    return (
        "System: You are a Goal Tracker Agent. Parse the user's goal, target, current savings and compute progress."
        " Provide a motivating update like 'You are 30% towards emergency fund'."
        " End with a fenced JSON block ```json {\"goal\": string, \"target_inr\": number, \"current_inr\": number, \"progress_pct\": number, \"tips\": [string]} ``` .\n\n"
        f"Insights: {insights}\nUser: {q}"
    )


def _decision_coach_compute_series(sip_inr: float, annual_rate: float, years: int) -> Tuple[List[int], List[float]]:
    return sip_projection(sip_inr, annual_rate, years)


def _decision_coach_agent(insights: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sip_inr = float(payload.get('sip') or 5000)
        rate = float(payload.get('rate') or 0.12)
        years = int(payload.get('years') or 10)
    except Exception:
        sip_inr, rate, years = 5000.0, 0.12, 10
    years_x, values = _decision_coach_compute_series(sip_inr, rate, years)
    summary = (
        f"If you invest ₹{int(sip_inr):,}/month for {years} years at {round(rate*100,2)}% expected return,"
        f" projected corpus ≈ ₹{int(values[-1] if values else 0):,}."
    )
    try:
        prompt = (
            "System: You are a Decision Coach Agent. Given a SIP, expected return, and years, explain the projection simply."
            " Include a brief risk note and suggest 2-3 fund categories."
            " Keep it bilingual (English + simple Hindi)."
            f"\n\nSIP: ₹{sip_inr}/month, Annual Return: {rate}, Years: {years}."
        )
        narrative = openrouter_invoke(prompt)
    except Exception:
        narrative = summary
    return {"text": narrative or summary, "series_years": years_x, "series_values": values}


def run_invest_agents_flow(agent: str, insights: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    state: Dict[str, Any] = {"agent": agent, "insights": insights, "payload": payload, "text": None, "cards": None}
    if StateGraph is None:
        # Fallback without LangGraph
        return _run_invest_agents_direct(agent, insights, payload)
    g = StateGraph(dict)

    def router(s: Dict[str, Any]) -> Dict[str, Any]:
        return s

    def education_node(s: Dict[str, Any]) -> Dict[str, Any]:
        q = str((s.get('payload') or {}).get('prompt', 'What is SIP?'))
        reply = None
        try:
            reply = openrouter_invoke(_education_agent_prompt(insights, q))
        except Exception:
            reply = None
        s['text'] = reply or "SIP (Systematic Investment Plan) means investing a fixed amount regularly in a mutual fund. It averages cost, builds discipline, and grows via compounding."
        blob = _extract_fenced_json(reply or '') if reply else None
        if isinstance(blob, dict) and isinstance(blob.get('cards'), list):
            s['cards'] = blob.get('cards')
        else:
            s['cards'] = [
                {"title": "What is SIP?", "subtitle": "Small steps → big corpus", "highlights": [
                    "Invest a fixed ₹ amount monthly",
                    "Averages market ups and downs",
                    "Harnesses compounding over time"
                ]}
            ]
        return s

    def analyzer_node(s: Dict[str, Any]) -> Dict[str, Any]:
        # Local analyzer without LLM to avoid blocking page loads
        ms = insights.get('monthly_spendings') or {}
        sv = insights.get('monthly_savings') or insights.get('savings_by_month') or {}
        saved_inr = 0.0
        ratio_change = 0.0
        try:
            if isinstance(sv, dict) and sv:
                last_key = sorted(sv.keys())[-1]
                saved_inr = float(sv.get(last_key) or 0)
        except Exception:
            saved_inr = 0.0
        # Expense ratio change (last vs previous): spend/income
        try:
            income = float(insights.get('income_per_month') or 0)
            if income and isinstance(ms, dict) and len(ms) >= 2:
                keys = sorted(ms.keys())
                cur = float(ms.get(keys[-1]) or 0)
                prev = float(ms.get(keys[-2]) or 0)
                cur_r = (cur / income) * 100.0
                prev_r = (prev / income) * 100.0
                ratio_change = round(cur_r - prev_r, 2)
        except Exception:
            ratio_change = 0.0
        s['metrics'] = {"saved_inr": int(saved_inr), "expense_ratio_change_pct": ratio_change}
        s['text'] = f"You saved ₹{int(saved_inr):,} this month. Expense ratio change: {ratio_change}% vs previous."
        return s

    def advisor_node(s: Dict[str, Any]) -> Dict[str, Any]:
        q = str((s.get('payload') or {}).get('prompt', 'long-term wealth'))
        reply = None
        try:
            reply = openrouter_invoke(_advisor_agent_prompt(insights, q))
        except Exception:
            reply = None
        s['text'] = reply or "Suggested: Start a ₹5,000/month SIP in an Index Fund for long-term wealth; keep an ELSS slice for tax-saving if relevant."
        blob = _extract_fenced_json(reply or '') if reply else None
        if isinstance(blob, dict):
            s['plan'] = blob
        else:
            # Default plan if LLM not available
            plan = {
                "company": "HDFC Mutual Fund",
                "product": "HDFC Index Fund - Nifty 50 Plan",
                "investment_type": "Index Fund",
                "amount_inr": 5000,
                "frequency": "month",
                "risk_score": 45,
                "why_this": "Low-cost, diversified, tracks Nifty 50; good for long-term core allocation.",
                "expected_return_pct": 11,
                "lock_in_years": 0,
                "duration_years": 10,
            }
            s['plan'] = plan
        return s

    def goal_node(s: Dict[str, Any]) -> Dict[str, Any]:
        # Local goal extraction without LLM for speed
        q = str((s.get('payload') or {}).get('prompt', 'Emergency fund target ₹100000, current ₹30000'))
        rupee = re.compile(r"(?:₹|INR|Rs\.?)[\s]*([\d,]+)")
        nums = [int(n.replace(',', '')) for n in rupee.findall(q)]
        target = nums[0] if nums else 100000
        current = nums[1] if len(nums) > 1 else 30000
        progress = int(min(100, max(0, (current/target*100) if target else 0)))
        tips = [
            "Automate a monthly transfer to your emergency fund",
            "Park funds in a liquid or ultra-short debt fund for safety + some yield"
        ]
        s['text'] = f"You’re {progress}% towards your emergency fund target. Target ₹{target:,}, Current ₹{current:,}."
        s['goal'] = {"goal": "Emergency Fund", "target_inr": target, "current_inr": current, "progress_pct": progress, "tips": tips}
        return s

    def coach_node(s: Dict[str, Any]) -> Dict[str, Any]:
        out = _decision_coach_agent(insights, s.get('payload') or {})
        s.update(out)
        return s

    g.add_node('router', router)
    g.add_node('education', education_node)
    g.add_node('analyzer', analyzer_node)
    g.add_node('advisor', advisor_node)
    g.add_node('goal', goal_node)
    g.add_node('coach', coach_node)
    g.add_edge('router', agent)
    g.add_edge(agent, END)
    g.set_entry_point('router')
    app = g.compile()
    result = app.invoke(state)
    return result


def _run_invest_agents_direct(agent: str, insights: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    if agent == 'education':
        q = str(payload.get('prompt', 'What is SIP?'))
        r = openrouter_invoke(_education_agent_prompt(insights, q))
        blob = _extract_fenced_json(r or '')
        cards = blob.get('cards') if isinstance(blob, dict) else None
        return {"agent": agent, "text": r, "cards": cards}
    if agent == 'analyzer':
        q = str(payload.get('prompt', 'overall'))
        r = openrouter_invoke(_analyzer_agent_prompt(insights, q))
        blob = _extract_fenced_json(r or '')
        return {"agent": agent, "text": r, "metrics": (blob or {}).get('metrics')}
    if agent == 'advisor':
        q = str(payload.get('prompt', 'long-term wealth'))
        r = openrouter_invoke(_advisor_agent_prompt(insights, q))
        return {"agent": agent, "text": r}
    if agent == 'goal':
        q = str(payload.get('prompt', 'Emergency fund target ₹100000'))
        r = openrouter_invoke(_goal_tracker_agent_prompt(insights, q))
        blob = _extract_fenced_json(r or '')
        return {"agent": agent, "text": r, "goal": blob}
    if agent == 'coach':
        out = _decision_coach_agent(insights, payload)
        out["agent"] = agent
        return out
    return {"agent": agent, "text": "Unsupported agent."}

# Curated Indian investment products (static seed; not real-time NAV). Estimates are illustrative.
INVEST_PRODUCTS: List[Dict[str, Any]] = [
    {"company": "HDFC Mutual Fund", "product": "HDFC Index Fund - Nifty 50 Plan", "category": "Index Fund", "min_sip": 100, "expected_return_pct": "10-12", "lock_in_years": 0, "durations": [1,3,5], "apply_link": "https://www.hdfcfund.com/", "logo": "/static/logos/hdfc-mf.svg", "top": True},
    {"company": "SBI Mutual Fund", "product": "SBI Nifty 50 ETF", "category": "Index Fund", "min_sip": 500, "expected_return_pct": "10-12", "lock_in_years": 0, "durations": [1,3,5], "apply_link": "https://www.sbimf.com/", "logo": "/static/logos/sbi-mf.svg", "top": True},
    {"company": "Nippon India", "product": "Nippon India Gold ETF", "category": "Gold ETF", "min_sip": 500, "expected_return_pct": "6-10", "lock_in_years": 0, "durations": [1,3,5], "apply_link": "https://mf.nipponindiamf.com/", "logo": "/static/logos/nippon-india.svg", "top": True},
    {"company": "ICICI Prudential", "product": "ICICI Prudential Liquid Fund", "category": "Liquid Fund", "min_sip": 100, "expected_return_pct": "5-7", "lock_in_years": 0, "durations": [1,3,5], "apply_link": "https://www.iciciprumf.com/", "logo": "/static/logos/icici-pru.svg", "top": True},
    {"company": "HDFC Mutual Fund", "product": "HDFC Liquid Fund", "category": "Liquid Fund", "min_sip": 100, "expected_return_pct": "5-7", "lock_in_years": 0, "durations": [1,3,5], "apply_link": "https://www.hdfcfund.com/", "logo": "/static/logos/hdfc-mf.svg"},
    {"company": "Axis Mutual Fund", "product": "Axis Long Term Equity Fund", "category": "ELSS", "min_sip": 500, "expected_return_pct": "10-14", "lock_in_years": 3, "durations": [3,5], "apply_link": "https://www.axismf.com/", "logo": "/static/logos/axis-mf.svg", "top": True},
    {"company": "Mirae Asset", "product": "Mirae Asset Large Cap Fund", "category": "Large Cap Equity", "min_sip": 500, "expected_return_pct": "10-13", "lock_in_years": 0, "durations": [3,5], "apply_link": "https://www.miraeassetmf.co.in/", "logo": "/static/logos/mirae-asset.svg"},
    {"company": "Kotak Mutual Fund", "product": "Kotak Corporate Bond Fund", "category": "Debt Fund", "min_sip": 100, "expected_return_pct": "7-9", "lock_in_years": 0, "durations": [1,3,5], "apply_link": "https://www.kotakmf.com/", "logo": "/static/logos/kotak-mf.svg", "top": True},
    {"company": "HDFC Mutual Fund", "product": "HDFC Silver ETF", "category": "Silver ETF", "min_sip": 500, "expected_return_pct": "6-10", "lock_in_years": 0, "durations": [1,3,5], "apply_link": "https://www.hdfcfund.com/", "logo": "/static/logos/hdfc-mf.svg"},
    {"company": "Nippon India", "product": "Nippon India Silver ETF", "category": "Silver ETF", "min_sip": 500, "expected_return_pct": "6-10", "lock_in_years": 0, "durations": [1,3,5], "apply_link": "https://mf.nipponindiamf.com/", "logo": "/static/logos/nippon-india.svg"},
    {"company": "ICICI Prudential", "product": "ICICI Prudential Silver ETF", "category": "Silver ETF", "min_sip": 500, "expected_return_pct": "6-10", "lock_in_years": 0, "durations": [1,3,5], "apply_link": "https://www.iciciprumf.com/", "logo": "/static/logos/icici-pru.svg"},
    {"company": "Government of India", "product": "Public Provident Fund (PPF)", "category": "PPF", "min_sip": 500, "expected_return_pct": "7-8", "lock_in_years": 15, "durations": [5,15], "apply_link": "https://www.indiapost.gov.in/", "logo": "/static/logos/ppf.svg", "top": True},
    {"company": "Pension Fund Reg. and Dev. Authority", "product": "National Pension System (NPS)", "category": "NPS", "min_sip": 500, "expected_return_pct": "8-12", "lock_in_years": 10, "durations": [10,20], "apply_link": "https://www.npscra.nsdl.co.in/", "logo": "/static/logos/nps.svg"},
    {"company": "Leading Banks", "product": "Fixed Deposit (FD)", "category": "Fixed Deposit", "min_sip": 1000, "expected_return_pct": "6-8", "lock_in_years": 0, "durations": [1,3,5], "apply_link": "https://www.sbi.co.in/", "logo": "/static/logos/fd.svg"},
]


def get_invest_products() -> List[Dict[str, Any]]:
    return INVEST_PRODUCTS


# Curated Health Insurance catalogs (private + government)
HEALTH_PRIVATE_PLANS: List[Dict[str, Any]] = [
    {"brand":"ACKO", "plan":"ACKO Family Health Plan", "type":"Family Floater", "cover_amount_inr": 500000, "premium_inr": 9000, "waiting_period_months": 24, "apply_link":"https://www.acko.com/health-insurance/", "logo":"https://logo.clearbit.com/acko.com"},
    {"brand":"Niva Bupa", "plan":"Niva Bupa ReAssure 2.0", "type":"Family Floater", "cover_amount_inr": 1000000, "premium_inr": 14000, "waiting_period_months": 36, "apply_link":"https://www.nivabupa.com/health-insurance.html", "logo":"https://logo.clearbit.com/nivabupa.com"},
    {"brand":"Care", "plan":"Care Supreme", "type":"Individual", "cover_amount_inr": 500000, "premium_inr": 8500, "waiting_period_months": 24, "apply_link":"https://www.careinsurance.com/health-insurance/", "logo":"https://logo.clearbit.com/careinsurance.com"},
    {"brand":"Star Health", "plan":"Star Family Health Optima", "type":"Family Floater", "cover_amount_inr": 500000, "premium_inr": 12000, "waiting_period_months": 36, "apply_link":"https://www.starhealth.in/", "logo":"https://logo.clearbit.com/starhealth.in"},
    {"brand":"ICICI Lombard", "plan":"Complete Health Insurance", "type":"Individual", "cover_amount_inr": 500000, "premium_inr": 11000, "waiting_period_months": 24, "apply_link":"https://www.icicilombard.com/health-insurance", "logo":"https://logo.clearbit.com/icicilombard.com"},
    {"brand":"HDFC ERGO", "plan":"Optima Secure", "type":"Family Floater", "cover_amount_inr": 1000000, "premium_inr": 16000, "waiting_period_months": 36, "apply_link":"https://www.hdfcergo.com/health-insurance", "logo":"https://logo.clearbit.com/hdfcergo.com"},
    {"brand":"TATA AIG", "plan":"MediCare Protect", "type":"Individual", "cover_amount_inr": 500000, "premium_inr": 10000, "waiting_period_months": 24, "apply_link":"https://www.tataaig.com/health-insurance", "logo":"https://logo.clearbit.com/tataaig.com"},
    {"brand":"Bajaj Allianz", "plan":"Health Guard", "type":"Family Floater", "cover_amount_inr": 500000, "premium_inr": 11500, "waiting_period_months": 24, "apply_link":"https://www.bajajallianz.com/health-insurance.html", "logo":"https://logo.clearbit.com/bajajallianz.com"},
    {"brand":"HDFC ERGO", "plan":"Maternity & Newborn Add-on", "type":"Maternity", "cover_amount_inr": 50000, "premium_inr": 2500, "waiting_period_months": 24, "apply_link":"https://www.hdfcergo.com/health-insurance", "logo":"https://logo.clearbit.com/hdfcergo.com"},
    {"brand":"Care", "plan":"Critical Illness", "type":"Critical Illness", "cover_amount_inr": 1000000, "premium_inr": 3500, "waiting_period_months": 90, "apply_link":"https://www.careinsurance.com/critical-illness-insurance", "logo":"https://logo.clearbit.com/careinsurance.com"},
    {"brand":"ACKO", "plan":"ACKO Super Top-Up", "type":"Super Top-Up", "cover_amount_inr": 1000000, "premium_inr": 3000, "waiting_period_months": 12, "apply_link":"https://www.acko.com/health-insurance/super-top-up/", "logo":"https://logo.clearbit.com/acko.com"},
]

HEALTH_GOVT_SCHEMES: List[Dict[str, Any]] = [
    {"brand":"Ayushman Bharat (PM-JAY)", "plan":"PM-JAY - ₹5L Coverage", "type":"Government", "cover_amount_inr": 500000, "premium_inr": 0, "waiting_period_months": 0, "apply_link":"https://pmjay.gov.in/", "logo":"https://logo.clearbit.com/pmjay.gov.in"},
    {"brand":"ESI", "plan":"Employees' State Insurance", "type":"Government", "cover_amount_inr": 0, "premium_inr": 0, "waiting_period_months": 0, "apply_link":"https://www.esic.nic.in/", "logo":"https://logo.clearbit.com/esic.nic.in"},
    {"brand":"CGHS", "plan":"Central Government Health Scheme", "type":"Government", "cover_amount_inr": 0, "premium_inr": 0, "waiting_period_months": 0, "apply_link":"https://cghs.gov.in/", "logo":"https://logo.clearbit.com/cghs.gov.in"},
    {"brand":"Rajasthan", "plan":"Mukhyamantri Chiranjeevi Yojana", "type":"Government", "cover_amount_inr": 1000000, "premium_inr": 0, "waiting_period_months": 0, "apply_link":"https://chiranjeevi.rajasthan.gov.in/", "logo":"https://logo.clearbit.com/rajasthan.gov.in"},
]


def get_health_catalog() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    return HEALTH_PRIVATE_PLANS, HEALTH_GOVT_SCHEMES


def recommend_health_plans(adults: int, kids: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    priv, govt = get_health_catalog()
    rec: List[Dict[str, Any]] = []
    if adults >= 2 or kids > 0:
        rec += [p for p in priv if p.get('type') in ('Family Floater', 'Super Top-Up', 'Maternity')]
    if adults == 1 and kids == 0:
        rec += [p for p in priv if p.get('type') in ('Individual', 'Critical Illness')]
    # If no match, show a balanced set
    if not rec:
        rec = [p for p in priv if p.get('type') in ('Family Floater','Individual')]
    # Limit to 6 for clarity
    rec = rec[:6]
    return rec, govt
