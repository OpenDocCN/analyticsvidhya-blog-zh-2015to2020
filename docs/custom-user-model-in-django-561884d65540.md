# Django 中的自定义用户模型

> 原文：<https://medium.com/analytics-vidhya/custom-user-model-in-django-561884d65540?source=collection_archive---------12----------------------->

上周六，我阅读了 Django 用户模型字段的文档。我都不记得我在找什么了。但是，当我阅读了用户名字段的描述后，我意识到我需要重写整个项目。

![](img/44a77f8f180f84597ba42d1c41a4d2ea.png)

照片由 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的 [NeONBRAND](https://unsplash.com/@neonbrand?utm_source=medium&utm_medium=referral) 拍摄

如你所知，Django 有一个内置的用户模型，这是默认选择。我在我的项目中使用了它，但该项目需要通过电子邮件进行认证。我们的时间有限，所以我决定将电子邮件保存到用户名字段。丑陋，但迅速和作品。不要这样做。

来自 Django doc:

*用户名*

*必填。150 个字符或更少。用户名可以包含字母数字、* `*_*` *、* `*@*` *、* `*+*` *、* `*.*` *和* `*-*` *字符。*

[](https://docs.djangoproject.com/) [## 姜戈文件|姜戈文件|姜戈

### 你需要知道的关于姜戈的一切。你是刚接触 Django 还是编程？这是开始的地方！拥有…

docs.djangoproject.com](https://docs.djangoproject.com/) 

原来用户名字段只支持有限的字符集，它不包含电子邮件中所有可能的字符。

在这种情况下，有两种解决方案。您可以使用自定义用户模型，也可以重写注册和身份验证后端以使用标准用户模型，并接受电子邮件作为登录。

我决定走第一条路，以便能够在未来对用户模型进行一些更改。

这里应该注意的是，在现有项目中迁移到定制用户模型是非常复杂的。我的项目目前还不大。所以我可以生成一个新的。

另一件事是用户模型类应该在 models.py 文件中并且只在那里。如果您将它移动到 models 文件夹并存储在 user.py 文件中，您将收到一个错误:

```
ModuleNotFoundError: No module named 'myappt.CustomUser'
```

我的模型. py:

```
from django.db import models
from django.contrib.auth.models import AbstractUser
from django.contrib.auth.models import BaseUserManager

class AppUserManager(BaseUserManager):
    def create_user(self, email, password, username, **extra_fields):
        if not email:
            raise ValueError('The Email must be set')
        if not username:
            raise ValueError('The Username must be set')
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save()
        return user

    def create_superuser(self, email, password, username, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        extra_fields.setdefault('is_active', True)

        if extra_fields.get('is_staff') is not True:
            raise ValueError('Superuser must have is_staff=True.')
        if extra_fields.get('is_superuser') is not True:
            raise ValueError('Superuser must have is_superuser=True.')
        return self.create_user(email, password, username, **extra_fields)

class AppUser(AbstractUser):
    email = models.EmailField(unique=True, blank=False)
    username = models.CharField(max_length=250, blank=True)
    is_staff = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    is_superuser = models.BooleanField(default=False)

    REQUIRED_FIELDS = ['username']
    USERNAME_FIELD = 'email'
    objects = AppUserManager()

    def __str__(self):
        return self.email
```

不要忘记将该字符串添加到 settings.py:

```
# Custom User model
AUTH_USER_MODEL = 'app.AppUser'
```

如果您想从管理面板管理您的用户，您必须向 admin.py 添加一些代码:

```
from .models import AppUser

class UserCreationForm(forms.ModelForm):
    *"""A form for creating new users. Includes all the required
    fields, plus a repeated password."""* password1 = forms.CharField(label='Password', widget=forms.PasswordInput)
    password2 = forms.CharField(label='Password confirmation', widget=forms.PasswordInput)

    class Meta:
        model = AppUser
        fields = ('email', 'username')

    def clean_password2(self):
        # Check that the two password entries match
        password1 = self.cleaned_data.get("password1")
        password2 = self.cleaned_data.get("password2")
        if password1 and password2 and password1 != password2:
            raise forms.ValidationError("Passwords don't match")
        return password2

    def save(self, commit=True):
        # Save the provided password in hashed format
        user = super().save(commit=False)
        user.set_password(self.cleaned_data["password1"])
        if commit:
            user.save()
        return user

class UserChangeForm(forms.ModelForm):
    *"""A form for updating users. Includes all the fields on
    the user, but replaces the password field with admin's
    password hash display field.
    """* password = ReadOnlyPasswordHashField()

    class Meta:
        model = AppUser
        fields = ('email', 'password', 'username', 'is_active', 'is_staff')

    def clean_password(self):
        # Regardless of what the user provides, return the initial value.
        # This is done here, rather than on the field, because the
        # field does not have access to the initial value
        return self.initial["password"]

class UserAdmin(BaseUserAdmin):
    # The forms to add and change user instances
    form = UserChangeForm
    add_form = UserCreationForm

    # The fields to be used in displaying the User model.
    # These override the definitions on the base UserAdmin
    # that reference specific fields on auth.User.
    list_display = ('email', 'username', 'is_staff')
    list_filter = ('is_staff',)
    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        ('Personal info', {'fields': ('username',)}),
        ('Permissions', {'fields': ('is_staff',)}),
    )
    # add_fieldsets is not a standard ModelAdmin attribute. UserAdmin
    # overrides get_fieldsets to use this attribute when creating a user.
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'username', 'password1', 'password2'),
        }),
    )
    search_fields = ('email',)
    ordering = ('email',)
    filter_horizontal = ()

admin.site.register(AppUser, UserAdmin)
admin.site.unregister(Group)
```

我希望这篇文章对你有用。