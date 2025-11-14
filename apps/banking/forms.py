from django import forms


class BankConnectForm(forms.Form):
    bank_name = forms.CharField(max_length=100)
    account_number = forms.CharField(max_length=50)
    ifsc_code = forms.CharField(max_length=20)
