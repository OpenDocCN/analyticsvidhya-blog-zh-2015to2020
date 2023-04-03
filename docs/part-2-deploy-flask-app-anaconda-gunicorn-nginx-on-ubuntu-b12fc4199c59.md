# 第 2 部分:在 Ubuntu 上部署 Flask App+Anaconda+guni corn+Nginx

> 原文：<https://medium.com/analytics-vidhya/part-2-deploy-flask-app-anaconda-gunicorn-nginx-on-ubuntu-b12fc4199c59?source=collection_archive---------9----------------------->

在之前的[文章](/@sarang0909.bds/part-1-deploy-flask-app-anaconda-gunicorn-nginx-on-ubuntu-4524014451b)中，我们已经设置了 web 应用程序和应用服务器。现在，让我们看看如何设置 Web 服务器。

在设置 nginx 之前，我们需要设置 gunicorn，以便它可以由 systemd 启动。这是 nginx 所必需的。

1.  Gunicorn by systemd

```
sudo vi /etc/systemd/system/myapp.service
```

向其中添加以下行:

```
[Unit]
Description=Gunicorn instance to serve My flask app
After=network.target
[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/my_code
Environment="PATH=/home/ubuntu/anaconda3/bin"
ExecStart=/home/ubuntu/anaconda3/envs/myapp/bin/gunicorn --workers 3 --bind unix:myapp.sock -m 007 **connector:app**
[Install]
WantedBy=multi-user.target
```

要启用链接:

```
sudo systemctl start myapp
sudo systemctl enable myapp
sudo systemctl status myapp
```

2.Nginx 设置

```
sudo apt-get install nginx
sudo usermod ubuntu -g www-data
```

创建 nginx 配置文件:

```
sudo vi /etc/nginx/sites-available/myappserver {
 listen 80;
 server_name x.x.x.x;
location / {
 include proxy_params;
 proxy_pass [http://unix:](http://unix:/home/ubuntu/kyc_validation/kycapp.sock)/home/ubuntu/my_code[/myapp.sock](http://unix:/home/ubuntu/kyc_validation/kycapp.sock);
 }
}
```

现在，创建符号链接

```
sudo ln -s /etc/nginx/sites-available/myapp /etc/nginx/sites-enabled
```

测试应用:

```
sudo systemctl start nginx
```

应用程序将运行在 [http://localhost:5000](http://localhost:5000)

3.监控和日志

要启用您的应用程序日志，请将以下代码添加到 flask 应用程序代码中:

```
import logginglogging.basicConfig(filename=APP_ROOT+'/'+'execution_log.log', filemode='a+', format=' [%(filename)s:%(lineno)s:%(funcName)s()]- %(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

gunicorn_logger = logging.getLogger('gunicorn.error')
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(gunicorn_logger.level)
```

启动和停止服务器的命令:

```
Gunicorn:
ps -ef|grep gunicorn    
sudo systemctl enable myapp
sudo systemctl status myapp
sudo systemctl start myapp
sudo systemctl stop myappNginx:
ps -ef | grep nginx
sudo systemctl status nginx.service
sudo systemctl start nginx 
sudo systemctl stop nginx 
sudo systemctl restart nginxsudo less /var/log/nginx/error.log: checks the Nginx error logs.
sudo less /var/log/nginx/access.log: checks the Nginx access logs.
sudo journalctl -u nginx: checks the Nginx process logs.
sudo journalctl -u myapp: checks your Flask app’s Gunicorn logs.
```

如果你喜欢这篇文章或有任何建议/意见，请在下面分享！

[领英](https://www.linkedin.com/in/sarang-mete-6797065a/)