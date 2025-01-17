from django.urls import path, include
from . import views 
from django.contrib import admin

app_name = 'sports' 
urlpatterns = [path('admin/', admin.site.urls),
                path('test/', views.index), 
                path('test/', views.test, name='test'),
                path('testtest/', views.testtest, name='testtest'),] 
