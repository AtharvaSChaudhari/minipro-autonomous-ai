from __future__ import annotations
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from apps.accounts.models import APIToken
from .services import (
    fetch_insights_via_api,
    ddg_health_insurance_results,
    health_agent_reply,
    get_health_catalog,
    recommend_health_plans,
    NBFC_LIST,
    loan_agent_reply,
    invest_agent_reply,
    sip_projection,
    get_invest_products,
    run_invest_agents_flow,
    education_cards_from_insights,
    advisor_plan_from_insights,
)
import secrets
import re
import json
import os


def _ensure_token(user) -> str:
    token_obj = getattr(user, 'api_token', None)
    if token_obj is None:
        token_obj = APIToken.objects.create(user=user, token=secrets.token_urlsafe(48))
    return token_obj.token


@login_required
def health_agent(request):
    token = _ensure_token(request.user)
    insights = {}
    if os.getenv('DISABLE_REMOTE_INSIGHTS', '0') != '1':
        try:
            insights = fetch_insights_via_api(token)
        except Exception as e:
            messages.warning(request, f"Could not fetch insights from API: {e}")

    results = ddg_health_insurance_results(limit=5)
    # Family inputs
    default_adults = 2
    default_kids = 1
    adults = int(request.POST.get('adults', default_adults) or default_adults)
    kids = int(request.POST.get('kids', default_kids) or default_kids)

    # Catalog and recommendations
    private_recs, govt_recs = recommend_health_plans(adults, kids)

    reply = None
    health_cards = None
    form_type = request.POST.get('form_type', '')
    if request.method == 'POST' and form_type == 'chat':
        prompt = request.POST.get('prompt', '').strip()
        if prompt:
            try:
                reply = health_agent_reply(insights, prompt)
                if reply:
                    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", reply)
                    if m:
                        try:
                            blob = json.loads(m.group(1))
                            if isinstance(blob, dict):
                                cards = blob.get('cards')
                                if isinstance(cards, list):
                                    # Normalize card fields
                                    norm = []
                                    for c in cards:
                                        if isinstance(c, dict) and c.get('title'):
                                            norm.append({
                                                'title': c.get('title'),
                                                'subtitle': c.get('subtitle'),
                                                'highlights': c.get('highlights') if isinstance(c.get('highlights'), list) else [],
                                                'cta_label': c.get('cta_label'),
                                                'cta_url': c.get('cta_url'),
                                            })
                                    health_cards = norm or None
                        except Exception:
                            health_cards = None
            except Exception as e:
                messages.error(request, f"Agent error: {e}")
    return render(request, 'agents/health.html', {
        'insights': insights,
        'results': results,
        'adults': adults,
        'kids': kids,
        'private_recs': private_recs,
        'govt_recs': govt_recs,
        'reply': reply,
        'health_cards': health_cards,
    })


@login_required
def loan_agent(request):
    token = _ensure_token(request.user)
    insights = {}
    try:
        insights = fetch_insights_via_api(token)
    except Exception as e:
        messages.warning(request, f"Could not fetch insights from API: {e}")

    reply = None
    if request.method == 'POST':
        prompt = request.POST.get('prompt', '').strip()
        if prompt:
            try:
                reply = loan_agent_reply(insights, prompt)
            except Exception as e:
                messages.error(request, f"Agent error: {e}")
    return render(request, 'agents/loan.html', {
        'insights': insights,
        'nbfcs': NBFC_LIST,
        'reply': reply,
    })


@login_required
def invest_agent(request):
    token = _ensure_token(request.user)
    insights = {}
    if os.getenv('DISABLE_REMOTE_INSIGHTS', '0') != '1':
        try:
            insights = fetch_insights_via_api(token)
        except Exception as e:
            messages.warning(request, f"Could not fetch insights from API: {e}")

    reply = None
    sip_monthly = float(request.POST.get('sip', '5000') or 5000)
    sip_rate = float(request.POST.get('rate', '0.12') or 0.12)
    sip_years = int(request.POST.get('years', '10') or 10)
    years_x, values = sip_projection(sip_monthly, sip_rate, sip_years)
    products = get_invest_products()
    parsed_plan = None
    plan_logo = None
    plan_apply = None
    projected_corpus = values[-1] if values else 0

    # Multi-agent (autonomous)
    education_cards = None
    education_text = None
    analyzer_text = None
    analyzer_metrics = None
    advisor_text = None
    advisor_plan = None
    advisor_logo = None
    advisor_apply = None
    goal_data = None

    do_llm = os.getenv('ENABLE_LLM_ON_PAGELOAD', '0') == '1'
    if do_llm:
        try:
            edu_result = run_invest_agents_flow('education', insights, {
                'prompt': 'What is SIP? Explain with examples for Indian investors.'
            })
            education_text = edu_result.get('text')
            cards = edu_result.get('cards')
            if isinstance(cards, list):
                education_cards = cards
        except Exception:
            pass
    else:
        education_cards = education_cards_from_insights(insights)

    try:
        analyzer_result = run_invest_agents_flow('analyzer', insights, {'prompt': 'overall'})
        analyzer_text = analyzer_result.get('text')
        analyzer_metrics = analyzer_result.get('metrics')
    except Exception:
        pass

    if do_llm:
        try:
            advisor_result = run_invest_agents_flow('advisor', insights, {'prompt': 'Plan for long-term wealth with moderate risk.'})
            advisor_text = advisor_result.get('text')
            plan = advisor_result.get('plan')
            if isinstance(plan, dict):
                advisor_plan = plan
        except Exception:
            advisor_plan = None
    if not advisor_plan:
        advisor_plan = advisor_plan_from_insights(insights)
    # Map logo/apply for advisor_plan if possible
    try:
        pc = str(advisor_plan.get('company', '')).strip().lower()
        pp = str(advisor_plan.get('product', '')).strip().lower()
        for p in products:
            if (pc and str(p.get('company','')).strip().lower() == pc) or (
                pp and str(p.get('product','')).strip().lower() == pp
            ):
                advisor_logo = p.get('logo')
                advisor_apply = p.get('apply_link')
                break
    except Exception:
        pass

    try:
        ms = insights.get('monthly_spendings') or {}
        avg_spend = 0.0
        if isinstance(ms, dict) and ms:
            total = 0.0
            count = 0
            for v in ms.values():
                try:
                    total += float(v)
                    count += 1
                except Exception:
                    continue
            if count:
                avg_spend = total / count
        if not avg_spend:
            avg_spend = 25000.0
        target = round(max(50000.0, 6 * avg_spend), 0)
        sdict = insights.get('savings_by_month') or insights.get('monthly_savings') or {}
        current = 0.0
        if isinstance(sdict, dict) and sdict:
            try:
                last_key = sorted(sdict.keys())[-1]
                current = float(sdict.get(last_key) or 0)
            except Exception:
                try:
                    current = float(list(sdict.values())[-1])
                except Exception:
                    current = 0.0
        if not current:
            try:
                current = float(insights.get('savings_total') or 0)
            except Exception:
                current = 0.0
        prompt_goal = f"Emergency fund target ₹{int(target)}, current ₹{int(current)}"
        goal_result = run_invest_agents_flow('goal', insights, {'prompt': prompt_goal})
        goal_data = goal_result.get('goal') or {
            'goal': 'Emergency Fund',
            'target_inr': int(target),
            'current_inr': int(current),
            'progress_pct': int(min(100, max(0, (current/target*100) if target else 0))),
            'tips': [
                'Automate a monthly transfer to your emergency fund',
                'Use windfalls (bonuses, refunds) to boost progress'
            ]
        }
    except Exception:
        pass

    if request.method == 'POST':
        prompt = request.POST.get('prompt', '').strip()
        if prompt:
            try:
                reply = invest_agent_reply(insights, prompt)
                if reply:
                    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", reply)
                    if m:
                        try:
                            parsed_plan = json.loads(m.group(1))
                            if isinstance(parsed_plan, dict):
                                pc = str(parsed_plan.get('company', '')).strip().lower()
                                pp = str(parsed_plan.get('product', '')).strip().lower()
                                for p in products:
                                    if pc and str(p.get('company','')).strip().lower() == pc or (
                                        pp and str(p.get('product','')).strip().lower() == pp
                                    ):
                                        plan_logo = p.get('logo')
                                        plan_apply = p.get('apply_link')
                                        break
                        except Exception:
                            parsed_plan = None
            except Exception as e:
                messages.error(request, f"Agent error: {e}")

    return render(request, 'agents/invest.html', {
        'insights': insights,
        'sip_monthly': sip_monthly,
        'sip_rate': sip_rate,
        'sip_years': sip_years,
        'years_x': years_x,
        'values': values,
        'reply': reply,
        'parsed_plan': parsed_plan,
        'products': products,
        'plan_logo': plan_logo,
        'plan_apply': plan_apply,
        'projected_corpus': projected_corpus,
        # Multi-agent context
        'education_cards': education_cards,
        'education_text': education_text,
        'analyzer_text': analyzer_text,
        'analyzer_metrics': analyzer_metrics,
        'advisor_text': advisor_text,
        'advisor_plan': advisor_plan,
        'advisor_logo': advisor_logo,
        'advisor_apply': advisor_apply,
        'goal_data': goal_data,
    })
