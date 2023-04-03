# 如何在 SQLite3 中使用 Flask-Login

> 原文：<https://medium.com/analytics-vidhya/how-to-use-flask-login-with-sqlite3-9891b3248324?source=collection_archive---------0----------------------->

![](img/e9c2f8a749313ca1a0d050ea4e1d67b5.png)

Python、Flask 和 SQLite3

大家好！！！

我看过很多关于如何用 **SQLAlchemy** 使用 Flask-Login 的教程。另一方面，我们也可以通过 **SQLite3 数据库**使用 Flask-Login。

其中 SQLite3 是比 SQLAlchemy 快 3 倍的 T4。

在这里，我将展示如何使用 SQLite3 数据库与 Flask-Login，这是我的项目的一部分，我已经向你展示。

重要说明我在这个项目中使用了**树莓 Pi 3** 。

> **首先，我们安装所有必要的软件开始运行，**

1.  创建一个 Python 虚拟环境。

```
sudo apt**-**get install python**-**virtualenv #Install Python virtual-envcd /var/ # Navigate to var directorymkdir www # Create directory named "www"cd wwwmkdir flaskcd flaskvirtualenv flask-sqlite3 #Create the virtual-env "flask-sqlite3" . flask-sqlite3/bin/activate # To activate virtual-envdeactivate # To deactivate virtual-env
```

2.安装 Flask，Flask-Login & wtforms。

```
. flask-sqlite3/bin/activatepip install flask # Install Flaskpip install flask-login # Install flask-loginpip install flask-wtf # Install flask-wtfpip install WTForms # Install WTForms
```

3.安装 SQLite3。

```
apt-get install sqlite3 # Install sqlite3sqlite3 login.db # Create a new sqlite database named "login"
```

“我使用 **Nginx** 作为我的网络服务器，使用 **Uwsgi** 作为我的应用程序网关”

> **设置 SQLite3**

```
sqlite3 login.db
```

现在我们将创建一个名为“login”的表来存储数据库中的所有登录数据。

```
create table login(
user_id NUMBER PRIMARY KEY AUTOINCREMENT,
email text not null,
password text not null);
```

通过使用**自动增量**，开发者不必在插入细节时跟踪 id。它会自动为每个注册用户分配 user _ ids。

**“使用注册表将登录数据插入 log in . db”**

我假设您已经在登录数据中插入了一些数据

```
insert into login values('xyz@gmail.com','XYZ123abc')
```

> **flask_login.py**

```
nano flask.py # Create a new python file named "flask" .  .  . from flask import Flask
from flask import render_template, url_for, flash, request, redirect, Response
import sqlite3
from flask_login import LoginManager, UserMixin, login_required, login_user, logout_user, current_user
from forms import LoginFormapp = Flask(__name__)
app.debug=Truelogin_manager = LoginManager(app)
login_manager.login_view = "login"class User(UserMixin):
    def __init__(self, id, email, password):
         self.id = unicode(id)
         self.email = email
         self.password = password
         self.authenticated = False def is_active(self):
         return self.is_active() def is_anonymous(self):
         return False def is_authenticated(self):
         return self.authenticated def is_active(self):
         return True def get_id(self):
         return self.id[@login_manager](http://twitter.com/login_manager).user_loader
def load_user(user_id):
   conn = sqlite3.connect('/var/www/flask/login.db')
   curs = conn.cursor()
   curs.execute("SELECT * from login where id = (?)",[user_id])
   lu = curs.fetchone()
   if lu is None:
      return None
   else:
      return User(int(lu[0]), lu[1], lu[2])[@app](http://twitter.com/app).route("/login", methods=['GET','POST'])
def login():
  if current_user.is_authenticated:
     return redirect(url_for('profile'))
  form = LoginForm()
  if form.validate_on_submit():
     conn = sqlite3.connect('/var/www/flask/login.db')
     curs = conn.cursor()
     curs.execute("SELECT * FROM login where email = (?)",    [form.email.data])
     user = list(curs.fetchone())
     Us = load_user(user[0])
     if form.email.data == Us.email and form.password.data == Us.password:
        login_user(Us, remember=form.remember.data)
        Umail = list({form.email.data})[0].split('@')[0]
        flash('Logged in successfully '+Umail)
        redirect(url_for('profile'))
     else:
        flash('Login Unsuccessfull.')
  return render_template('login.html',title='Login', form=form)if __name__ == "__main__":
  app.run(host='0.0.0.0',port=8080,threaded=True)
```

> **forms.py**

```
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError
import sqlite3class LoginForm(FlaskForm):
 email = StringField('Email',validators=[DataRequired(),Email()])
 password = PasswordField('Password',validators=[DataRequired()])
 remember = BooleanField('Remember Me')
 submit = SubmitField('Login') def validate_email(self, email):
    conn = sqlite3.connect('/var/www/flask/login.db')
    curs = conn.cursor()
    curs.execute("SELECT email FROM login where email = (?)",[email.data])
    valemail = curs.fetchone()
    if valemail is None:
      raise ValidationError('This Email ID is not registered. Please register before login')
```

现在我们使用 HTML 和 CSS 为登录页面设计模板。

```
mkdir templates # create a templates directory.cd templates
```

> **html_template.html**

```
<!DOCTYPE html>
<html lang="en">
<head><!-- Basic Page Needs
   -->
  <meta charset="utf-8">
  {% if title %}
   <title>{{ title }}</title>
  {% endif %}<!-- Mobile Specific Metas
   -->
  <meta name="viewport" content="width=device-width, initial-scale=1 shrink-to-fit=no"><!-- FONT
  -->    
  <link href="//fonts.googleapis.com/css?family=Raleway:400,300,600" rel="stylesheet" type="text/css"><link rel="stylesheet" href="../static/css/normalize.css">
  <link rel="stylesheet" href="../static/css/skeleton.css"><link rel="icon" type="image/png" href="../static/logo.png"></head>
<body><!-- Primary Page Layout
   -->
<div style="margin-top:5%; margin-left:5%">
 <div style="width:100%;">
  <div style="width: 80%; height: 100px; float: left;">
     {% with messages = get_flashed_messages(with_categories=True)%}
 {% if messages %}
  {% for category, message in messages %}
   <div class="alert alert-{{ category }}">
    <h5>{{ message }}<h5>
   </div>
  {% endfor %}
 {% endif %}
     {% endwith %}
     {% block info %}
     {% endblock %}
  </div>
 </div>
</div><!-- End Document
   -->
</body>
</html>
```

我使用了 **skeleton.css** 进行设计

参考[**http://getskeleton.com/**](http://getskeleton.com/)或[**https://github.com/dhg/Skeleton**](https://github.com/dhg/Skeleton)

> **login.html**

```
{% extends "html_template.html" %}
{% block info %}
 <form method='POST' action="">
  {{ form.hidden_tag() }}
  <fieldset class="form-group"> 
   <h1>Log In</h1>
   <div class="form-group">
    {{ form.email.label(class="form-control-label")}}
    {% if form.email.errors %}
     {{ form.email(class="form-control form-control-lg is-invalid") }}
     <div class="invalid-feedback">
      {% for error in form.email.errors %}
       <span>{{ error }}</span>
      {% endfor %}
     </div>
    {% else %}  
     {{ form.email(class="form-control form-control-lg")}}
    {% endif %}
   </div>
   <div class="form-group">
    {{ form.password.label(class="form-control-label")}}
    {% if form.password.errors %}
     {{ form.password(class="form-control form-control-lg is-invalid") }}
     <div class="invalid-feedback">
      {% for error in form.password.errors %}
       <span>{{ error }}</span>
      {% endfor %}
     </div>
    {% else %}
     {{ form.password(class="form-control form-control-lg")}}
    {% endif %}
   </div>
   <div class="form-group">
    {{ form.remember(class="form-check-input") }} 
    {{ form.remember.label(class="form-check-label") }}
   </div>
  </fieldset>
  <div class="form-group">
   {{ form.submit(class="btn btn-outline-info")}}
  </div>
  <a href="javascript:void(0)" onclick="location.href='/reset'">Forgot Password?</a>
 </form>
 <h6>Create an Account <a href="javascript:void(0)" onclick="location.href='/register'">Sign Up Now</a></h6>
{% endblock info %}
```

现在，运行 web 应用程序，

```
python flask_login.py
```

## 感谢所有觉得它有用的人