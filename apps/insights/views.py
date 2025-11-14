import json
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from apps.accounts.models import Profile
from apps.banking.models import BankConnection
from .ml_engine import compute_insights_from_url


@login_required
def dashboard(request):
    profile = Profile.objects.filter(user=request.user).first()
    if not profile:
        messages.info(request, 'Please complete your profile first.')
        return redirect('accounts:profile')

    bank = BankConnection.objects.filter(user=request.user).first()
    if not bank or not bank.csv_url:
        messages.info(request, 'Link a bank account to view insights.')
        return redirect('bank:connect')

    try:
        insights = compute_insights_from_url(bank.csv_url, float(profile.monthly_income))
    except Exception as e:
        messages.error(request, f'Failed to compute insights: {e}')
        show_flag = bool(request.session.pop('show_welcome', False))
        return render(request, 'insights/dashboard.html', {'insights': None, 'show_welcome': show_flag})

    show_flag = bool(request.session.pop('show_welcome', False))
    return render(request, 'insights/dashboard.html', {
        'insights': insights,
        'insights_json': json.dumps(insights),
        'show_welcome': show_flag,
    })
