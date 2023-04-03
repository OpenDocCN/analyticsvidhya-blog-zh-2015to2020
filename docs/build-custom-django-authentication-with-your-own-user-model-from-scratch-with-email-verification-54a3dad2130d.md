# 通过电子邮件验证，使用您自己的用户模型从头开始构建自定义 Django 身份验证

> 原文：<https://medium.com/analytics-vidhya/build-custom-django-authentication-with-your-own-user-model-from-scratch-with-email-verification-54a3dad2130d?source=collection_archive---------3----------------------->

![](img/9ea13a7740a8707246a90a11d43b3dce.png)

希望你做得很好！在这里，我将向您介绍如何在 django 中为您自己创建的用户模型构建一个定制的身份验证系统。假设您试图构建一个应用程序，其中您想要摆脱 django 的内置用户模型。你如何认证你自己的用户？开始吧！

首先创建一个项目。(这里我的项目名是 customAuth)。

> django-管理开始项目自定义授权

进入 *customAuth* 目录，创建一个应用 *authApp。*

> CD custom auth
> python manage . py startapp authApp

在 *customAuth/settings.py* 文件中注册您的应用程序

```
# Application definitionINSTALLED_APPS = [
    'authApp',
    ...
]
```

在您的 *authApp* 目录中，创建一个文件 *urls.py* 来创建对应于该应用程序的 URL，只在其中创建每个应用程序的 URL 总是一个好的做法。编辑您的 *authApp/urls.py* 如下。

```
from django.urls import pathapp_name = 'authApp'urlpatterns = []
```

现在将应用程序的 url 文件注册到项目的 url 中。打开 *customAuth/urls.py* 并添加以下行。

```
....
from django.urls import path, includeurlpatterns = [
    ....
    path('', include('authApp.urls')),
]
```

让我们通过迁移来形成我们的数据库。运行命令。

> python manage.py 迁移

现在我们可以专注于我们的主要目标了。在 *authApp/models.py* 中创建一个名为 *MyUser 的用户模型。(我已经为每个用户保留了唯一的用户名和电子邮件)*

```
class MyUser(models.Model):
 name = models.CharField(max_length=255)
 email = models.EmailField(max_length=500, unique=True)
 username = models.CharField(max_length=255, unique=True)
 password = models.CharField(max_length=255)def __str__(self):
  return self.username
```

应用迁移。

> python manage . py make migrations
> python manage . py 迁移

让我们为这个用户创建一个简单的注册和登录门户。为我们的 MyUser 模型创建注册和登录两个表单。打开 *authApp/forms.py* (如果不存在，创建一个文件 *forms.py* ) ，编辑如下。

```
from django import forms
from .models import MyUserpasswordInputWidget = {
 'password': forms.PasswordInput(),
}class RegisterForm(forms.ModelForm):
 class Meta:
  model = MyUser
  fields = '__all__'
  widgets = [passwordInputWidget]class LoginForm(forms.ModelForm):
 class Meta:
  model = MyUser
  fields = ['username', 'password']
  widgets = [passwordInputWidget]
```

让我们为我们的网站创建一些视图。打开 *authApp/views.py* ，编辑如下。然后创建对应于这些视图的 URL。

> authApp/views.py

```
from django.shortcuts import render
from .models import MyUser
from .forms import RegisterForm, LoginForm# Create your views here.def register(request):
 form = RegisterForm()
 return render(request, 'authApp/register.html', {'form': form})def login(request):
 form = LoginForm()
 return render(request, 'authApp/login.html', {'form': form})
```

> authApp/urls.py

```
...
from . import views
..urlpatterns = [
 path('register/', views.register, name='register'),
 path('login/', views.login, name='login'),
]
```

在*authApp/templates/authApp*目录下创建两个模板，分别命名为*register.html 和*，如下所示。

> register.html

```
<form method="post">
 {% csrf_token %}
 {{ form.as_p }}
 <button>Register</button>
 <div>Already have an account? <a href="{% url 'authApp:login' %}">Sign in</a>
</form>
```

> login.html

```
<form method="post">
 {% csrf_token %}
 {{ form.as_p }}
 <button>Login</button>
 <div>Don't have an account? <a href="{% url 'authApp:register' %}">Sign up</a>
</form>
```

现在我们已经为我们的网站创建了一个基本视图。打开您的终端/命令窗口，并在其中运行以下命令。

> python manage.py runserver

并在浏览器中打开 localhost:8000/register。你拿到登记表了吗？此外，在单击登录链接时，您将被重定向到登录页面。基本观点做好了。现在我们只剩下认证部分了。当有人注册时，registerform 被发布，它应该用发布的数据创建一个新的 MyUser 实例，如果用户名和电子邮件不是唯一的，那么我们将显示一个错误。所以让我们编辑 *views.py.*

```
def register(request):
 form = RegisterForm()
 success = None
 if request.method=='POST':
  if MyUser.objects.filter(username=request.POST['username']).exists():
   error = "This username is already taken"
   return render(request, 'authApp/register.html', {'form': form, 'error': error})
  if MyUser.objects.filter(email=request.POST['email']).exists():
   error = "This email is already taken"
   return render(request, 'authApp/register.html', {'form': form, 'error': error})
  form = RegisterForm(request.POST)
  new_user = form.save(commit=False)
  new_user.save()
  success = "New User Created Successfully !"
 return render(request, 'authApp/register.html', {'form': form, 'success': success})
```

编辑 register.html 如下。

```
{% if success %}
<div> {{ success }} </div>
{% endif %}{% if error %}
<div> {{ error }} </div>
{% endif %}<form method="post">
 ...
</form>
```

当您提交包含有效条目的注册表时，将会创建新用户。请随意注册一些用户，并尝试用相同的用户名和电子邮件创建帐户，并验证错误是否有效。现在，我们已经创建了用户注册系统，我们必须让用户登录到我们的系统。当用户使用正确的凭证提交登录表单时，找到用户对应的用户名并为该用户创建一个新会话。只要用户登录，会话就会为该用户运行。用户登录系统后，我们会将其重定向到一个欢迎页面(或*主页*页面),在这里我们会显示用户的信息。我们还将使主页仅在用户登录后才可见，也就是说，除非用户登录，否则无法进入主页。为此，我们必须创建自己的装饰函数，就像 *login_required 一样。*首先创建一个文件*authApp/templates/authApp/home . html*，编辑如下文件。

> authApp/views.py

```
from django.shortcuts import render, redirectdef login(request):
 form = LoginForm()
 if request.method=='POST':
  username = request.POST['username']
  password = request.POST['password']
  if MyUser.objects.filter(username=username, password=password).exists():
   user = MyUser.objects.get(username=username)
   request.session['user_id'] = user.id # This is a session variable and will remain existing as long as you don't delete this manually or clear your browser cache
   return redirect('authApp:home')
 return render(request, 'authApp/login.html', {'form': form})def get_user(request):
 return MyUser.objects.get(id=request.session['user_id'])def home(request):
 if 'user_id' in request.session:
  user = get_user(request)
  return render(request, 'authApp/home.html', {'user': user})
 else:
  return redirect('authApp:login')
```

> authApp/urls.py

```
urlpatterns = [
 ....
 path('', views.home, name='home'),
]
```

> auth app/templates/auth app/home . html

```
<div>
 Welcome {{ user.username }}
 <p> Name : {{ user.name }} </p>
 <p> Username : {{ user.username }} </p>
 <p> Email : {{ user.email }} </p>
</div>
```

现在转到 localhost:8000，您应该会被自动重定向到登录页面。尝试使用有效凭据登录。希望你现在可以进入主页了。但是对于大型 web 应用程序，将会有许多用户只能在登录后才能访问的网页，所以不要设置会话密钥是否存在的条件，让我们构建自己的装饰器。创建一个文件 *authApp/decorators.py* 并编辑以下文件。

> decorators.py

```
from .models import MyUser
from django.shortcuts import redirectdef user_login_required(function):
 def wrapper(request, login_url='authApp:login', *args, **kwargs):
  if not 'user_id' in request.session:
   return redirect(login_url)
  else:
   return function(request, *args, **kwargs)
 return wrapper
```

> views.py

```
from .decorators import user_login_required[@user_login_required](http://twitter.com/user_login_required)
def home(request):
 user = get_user(request)
 return render(request, 'authApp/home.html', {'user': user})
```

添加**注销**功能的时间到了。编辑 *views.py* 和 *urls.py.*

> views.py

```
def logout(request):
 if 'user_id' in request.session:
  del request.session['user_id'] # delete user session
 return redirect('authApp:login')
```

> urls.py

```
urlpatterns = [
 ...
 path('logout/', views.logout, name='logout'),
]
```

> auth app/templates/auth app/home . html

```
......
<div>To logout, <a href="{% url 'authApp:logout' %}">Click Here</a></div>
```

现在，当你进入主页后，尝试注销，你将被重定向到登录页面。请注意，在您注销后，除非您再次登录，否则无法访问主页。

***邮件验证***

让我们创建一些功能，使欺诈用户无法进入我们的系统。当用户注册时，我们会发送一些电子邮件验证码到他的邮箱。他只能用正确的验证码注册。我们将调用 ajax 来实时发送电子邮件，而无需刷新页面。在 register.html 导入 jquery，用邮件数据回复 *views.py* ，再次返回注册页面。您需要更改 *customAuth/settings.py* 才能访问邮件服务。

> settings.py

```
...

EMAIL_HOST = 'smtp.gmail.com' ## Email you want to use
EMAIL_PORT = 587 # For gmail, code is 587
EMAIL_HOST_USER = "<YOUR EMAIL ADDRESS>"
EMAIL_HOST_PASSWORD = "<YOUR EMAIL PASSWORD>"
EMAIL_USE_TLS = True
EMAIL_USE_SSL = False
```

> register.html

```
<script src="[https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js](https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js)"></script>{% if success %}
<div> {{ success }} </div>
{% endif %}{% if error %}
<div> {{ error }} </div>
{% endif %}<form method="post">
 {% csrf_token %}
 {{ form.as_p }}
 <input type='number' name='code' id='id_code' placeholder="Enter Verification Code.." style="display: none;">
 <button id='regBtn'>Register</button>
 <div>Already have an account? <a href="{% url 'authApp:login' %}">Sign in</a>
</form><script>
 $('#regBtn').on('click', function(e){
  if($("#id_code").css('display')=='none'){
   e.preventDefault();
   var dataToGo = $('#id_email').val();
   console.log(dataToGo);
   $.ajax({
           url: "{% url 'authApp:ajax_generate_code' %}",
           type: 'GET',
           data: dataToGo,
           success: function (data) {
               $('#id_code').css('display', 'block');
           },
           cache: false,
           contentType: false,
           processData: false
        });
  }
 })
</script>
```

> views.py

```
import random
from django.core.mail import EmailMultiAlternatives
from django.template.loader import render_to_string
from customAuth.settings import EMAIL_HOST_USER
from django.http import HttpResponsedef ajax_generate_code(request):
 print(request.GET)
 for x in request.GET:
  if x!='_':
   email = x## Generate Code and save it in a session
   request.session['code'] = random.randint(100000, 999999)## Send email Functionality
   text_content = "Your Email Verification Code is " + str(request.session['code'])
   msg = EmailMultiAlternatives('Verify Email', text_content, EMAIL_HOST_USER, [email])
   msg.send()return HttpResponse("success")def register(request):
 form = RegisterForm()
 success = None
 if request.method=='POST':
  if MyUser.objects.filter(username=request.POST['username']).exists():
   error = "This username is already taken"
   return render(request, 'authApp/register.html', {'form': form, 'error': error})
  if MyUser.objects.filter(email=request.POST['email']).exists():
   error = "This email is already taken"
   return render(request, 'authApp/register.html', {'form': form, 'error': error})## Check Verification Code
  if (not 'code' in request.POST) or (not 'code' in request.session) or (not request.POST['code']==str(request.session['code'])):
   error = "Invalid Verification Code"
   return render(request, 'authApp/register.html', {'form': form, 'error': error})## Safe to go
  form = RegisterForm(request.POST)
  new_user = form.save(commit=False)
  new_user.save()
  success = "New User Created Successfully !"
 return render(request, 'authApp/register.html', {'form': form, 'success': success})
```

> urls.py

```
...
urlpatterns = [
 ...
 path('ajax_generate_code/', views.ajax_generate_code, name='ajax_generate_code'),
]
```

转到 localhost:8000/register/并尝试创建一个帐户，在那里输入您的电子邮件地址，当您第一次单击注册按钮时，它会显示另一个电子邮件验证码输入框。检查你的电子邮件，你应该得到一个验证码。把它放在那里，你就可以走了！

编码快乐！