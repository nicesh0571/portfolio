from django.urls import path
from .views import *

app_name='v1'
urlpatterns = [
    path('', index, name='index'),
]