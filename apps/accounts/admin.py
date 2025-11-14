from django.contrib import admin
from .models import Profile, APIToken

@admin.register(Profile)
class ProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "phone", "monthly_income", "gig_platforms", "family_members")

@admin.register(APIToken)
class APITokenAdmin(admin.ModelAdmin):
    list_display = ("user", "token", "created_at")
