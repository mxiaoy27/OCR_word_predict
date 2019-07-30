'''

from django.conf.urls import url
from django.contrib import admin
from django.conf.urls import include
from webdev import views

urlpatterns = [

    url(r'^admin/', admin.site.urls),
    url(r'^login/', views.login),
    url(r'^predict/', views.predict, name='predict'),
    url(r'^upload/', include("webdev.urls"))
]
'''

from django.contrib import admin
from django.urls import path, include
from webdev import views
from django.conf.urls import url

urlpatterns = [
    path('admin/', admin.site.urls),
    path('index/', views.index),
    path('predict/', views.upload_file, name='predict'),
]
