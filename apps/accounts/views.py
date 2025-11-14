from django.shortcuts import render, redirect
from django.contrib.auth import login
from django.contrib.auth.views import LoginView
from django.contrib.auth.decorators import login_required
from .forms import SignUpForm, ProfileForm
from .models import Profile, APIToken
import secrets


class LoginViewCustom(LoginView):
    template_name = 'accounts/login.html'

    def form_valid(self, form):
        resp = super().form_valid(form)
        # Trigger welcome banner on next dashboard visit
        self.request.session['show_welcome'] = True
        return resp


def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            user.email = form.cleaned_data['email']
            user.first_name = form.cleaned_data['name']
            user.save()
            Profile.objects.create(
                user=user,
                phone=form.cleaned_data['phone'],
                monthly_income=form.cleaned_data['monthly_income'],
                gig_platforms=form.cleaned_data['gig_platforms'],
                family_members=form.cleaned_data['family_members'],
            )
            APIToken.objects.create(user=user, token=secrets.token_urlsafe(48))
            login(request, user)
            # Trigger welcome banner on next dashboard visit
            request.session['show_welcome'] = True
            return redirect('bank:connect')
    else:
        form = SignUpForm()
    return render(request, 'accounts/signup.html', {'form': form})


@login_required
def profile(request):
    profile_obj, _ = Profile.objects.get_or_create(user=request.user)
    if request.method == 'POST':
        form = ProfileForm(request.POST, instance=profile_obj)
        if form.is_valid():
            form.save()
            return redirect('insights:dashboard')
    else:
        form = ProfileForm(instance=profile_obj)
    return render(request, 'accounts/profile.html', {'form': form})
