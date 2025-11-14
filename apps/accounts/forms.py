from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Profile


class SignUpForm(UserCreationForm):
    name = forms.CharField(max_length=150, label="Name")
    email = forms.EmailField()
    phone = forms.CharField(max_length=20)
    monthly_income = forms.DecimalField(max_digits=12, decimal_places=2)
    gig_platforms = forms.IntegerField(min_value=0)
    family_members = forms.IntegerField(min_value=1)

    class Meta:
        model = User
        fields = ("username", "name", "email", "password1", "password2", "phone", "monthly_income", "gig_platforms", "family_members")


class ProfileForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ("phone", "monthly_income", "gig_platforms", "family_members")
