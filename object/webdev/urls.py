'''
from webdev import views
from django.conf.urls import url  # Django2.0

app_name = 'namespace'
urlpatterns = [
    url(r'^upload/', views.upload_file),
]
'''
from webdev import views
from django.urls import path

# urlpatterns = [
#     path('index/', views.index),
#     path('predict/', views.upload_file, name='predict'),
# ]