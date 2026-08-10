from django.urls import path
from . import views

app_name = 'diagnosis'


urlpatterns = [
    path("", views.home, name="home"),
    path("upload/", views.upload_image, name="upload"),
    path("result/<int:pk>/", views.result, name="result"),
    path("history/", views.history, name="history"),
    path("profile/", views.profile, name="profile"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("register/", views.register, name="register"),
    path("history/delete/<int:pk>/",views.delete_prediction,name="delete_prediction"),
    path("history/pdf/", views.download_pdf, name="download_pdf"),
]

