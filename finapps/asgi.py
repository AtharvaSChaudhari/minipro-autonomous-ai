import os
from django.core.asgi import get_asgi_application
from starlette.applications import Starlette
from starlette.routing import Mount

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'finapps.settings')

django_asgi_app = get_asgi_application()

# Import after Django setup
from api.app import fastapi_app  # noqa: E402

app = Starlette(routes=[
    Mount('/api', fastapi_app),
    Mount('/', django_asgi_app),
])
