from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import BankConnectForm
from .models import BankConnection
from .supabase_client import get_csv_link


@login_required
def connect(request):
    if request.method == 'POST':
        form = BankConnectForm(request.POST)
        if form.is_valid():
            bank_name = form.cleaned_data['bank_name']
            account_number = form.cleaned_data['account_number']
            ifsc_code = form.cleaned_data['ifsc_code']
            try:
                csv_url = get_csv_link(bank_name, account_number, ifsc_code)
            except Exception as e:
                messages.error(request, f"Failed to query Supabase: {e}")
                csv_url = None
            if not csv_url:
                messages.error(request, 'No CSV link found for the provided bank details.')
            else:
                BankConnection.objects.create(
                    user=request.user,
                    bank_name=bank_name,
                    account_number=account_number,
                    ifsc_code=ifsc_code,
                    csv_url=csv_url,
                )
                messages.success(request, 'Bank account linked successfully!')
                return redirect('insights:dashboard')
    else:
        form = BankConnectForm()
    return render(request, 'banking/add_bank.html', {'form': form})
