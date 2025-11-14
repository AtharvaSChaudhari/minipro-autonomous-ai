from django.db import models
from django.contrib.auth.models import User


class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    phone = models.CharField(max_length=20, blank=True)
    monthly_income = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    gig_platforms = models.PositiveIntegerField(default=0)
    family_members = models.PositiveIntegerField(default=1)

    def __str__(self):
        return f"Profile({self.user.username})"


class APIToken(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='api_token')
    token = models.CharField(max_length=128, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"APIToken({self.user.username})"
