from django.urls import path
from .import views
from django.conf.urls.static import static
from django.conf import settings
app_name='sample'

urlpatterns = [
    path('',views.home,name='home'),
    path('register',views.register,name="register"),
    path('login',views.login,name='login'),
    path('logout',views.logout,name='logout'),
    path(' ',views.header,name='header'),
    path('counter',views.counter,name='counter'),
    path('detection',views.detection,name='detection'),
    path('help',views.help,name='help'),

]
urlpatterns+=static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)

 