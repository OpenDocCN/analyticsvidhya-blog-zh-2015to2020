# 在 Heroku 免费层部署一个闪亮的应用程序

> 原文：<https://medium.com/analytics-vidhya/deploying-an-r-shiny-app-on-heroku-free-tier-b31003858b68?source=collection_archive---------5----------------------->

GitHub Actions 和 Heroku 使持续部署变得容易

![](img/1f5ca13046d31271ec5c2c91fa67d6b6.png)

这篇文章是在 Heroku 上部署闪亮应用的简短指南。假定熟悉 Docker、Shiny 和 GitHub。关于 Docker 的介绍，请参见 [*用 Docker*](/analytics-vidhya/deploying-a-shiny-flexdashboard-with-docker-cca338a10d12) *部署闪亮的 Flexdashboard。*

*这篇文章也发表在*[*https://www.r-bloggers.com/*](https://www.r-bloggers.com/)*上。*

## 这篇文章将向你展示

*   什么是 Heroku，你能免费得到什么
*   如何用 Docker 封装一个闪亮的应用程序
*   如何设置 GitHub 自动部署到 Heroku
*   如何为您部署的应用程序设置自定义域名

# Heroku 自由层

H [eroku](https://www.heroku.com/home) (我不隶属于它)是一个云平台服务，它提供了一种直观的方式来部署和管理企业级 web 应用。它可以让你的应用程序轻松地自动缩放，但不是在免费(和爱好)层，我们将在这里使用。可以通过 CLI 或 Heroku 仪表板管理应用程序。

Heroku 有太多的[插件](https://elements.heroku.com/addons)，例如用于登录或发送信息。此外，只需在 CLI 或部署说明文件“heroku.yml”中点击几下鼠标或输入几行代码，它就可以提供与应用程序共同部署的服务，如数据库。

使用新的 [heroku.ym](https://devcenter.heroku.com/articles/build-docker-images-heroku-yml) l 文件，可以部署多容器应用程序。但是这里我们不会使用 heroku.yml，因为部署将由 GitHub Actions 管理。

Heroku 应用程序运行在“dynos”中，dynos 是运行在 AWS 上的容器。使用免费层，您可以在每个 512 MB 内存的情况下运行最多 5 个 dyno(100，如果您验证)的应用程序，并获得 550 (1000，如果您验证)个 dyno 小时。在免费层，你的应用程序在 30 分钟后进入“睡眠模式”并需要被唤醒，这通常意味着你第一次请求的加载时间将会延长 10-30 秒。这种行为节省了你的空闲时间。如果你想避免这一点，给你的应用程序添加一个 GET 请求，这样应用程序就会以小于 30 分钟的间隔调用自己。我们将会看到这能持续多久。

我喜欢 Heroku 定价模式的原因是它是基于应用的。这意味着您可以在免费层测试后升级单个应用程序。在付费层，您可以获得各种好处，如自动缩放，无限的动态时间和自定义域的 SSL 加密。

# 闪亮的应用程序 Docker 图像

H eroku dynos 支持一些开箱即用的语言，如 Python 和附带的[构建包](https://elements.heroku.com/buildpacks)。在 [github](https://github.com/virtualstaticvoid/heroku-buildpack-r) 上有[非官方的](https://davidquartey.medium.com/how-i-installed-r-on-heroku-ff8286233d2c)R 的 buildpacks。然而，这里我们将使用 Docker 容器在 dyno 中运行。此外，将闪亮的应用程序放在 Docker 容器中有各种优势，例如正确管理依赖性和跨系统的良好可移植性。

在这个例子中，部署了一个 R Shiny 应用程序，它提供地址到经度和纬度的批量地理编码。在回购中你可以找到并克隆闪亮的应用，Dockerfile 和 GitHub Actions https://github.com/timosch29/geocoding_shiny:[YAML](https://github.com/timosch29/geocoding_shiny)

部署的应用程序可以在[geocoding.timschendzielorz.com](https://geocoding.timschendzielorz.com)找到。

对于 Heroku 上的部署，您必须稍微修改 Dockerfile 指令。

```
# Base image https://hub.docker.com/u/rocker/                       FROM rocker/shiny-verse:4.0.3LABEL author="Tim M.Schendzielorz docker@timschendzielorz.com"# system libraries of general use                       
# install debian packages                       
RUN apt-get update -qq && apt-get -y --no-install-recommends install \ 
    libxml2-dev \
    libcairo2-dev \
    libpq-dev \ 
    libssh2-1-dev \
    libcurl4-openssl-dev \
    libssl-dev# update system libraries
RUN apt-get update && \ 
    apt-get upgrade -y && \  
    apt-get clean# copy necessary files from app folder
# Shiny app 
COPY /shiny_geocode ./app                       
# renv.lock file
COPY /renv.lock ./renv.lock# install renv & restore packages                       
RUN Rscript -e 'install.packages("renv")'
RUN Rscript -e 'renv::restore()'# remove install files                       
RUN rm -rf /var/lib/apt/lists/*# make all app files readable, gives rwe permisssion (solves issue when dev in Windows, but building in Ubuntu)                       RUN chmod -R 755 /app# expose port (for local deployment only)                       EXPOSE 3838# set non-root                       
RUN useradd shiny_user
USER shiny_user# run app on container start (use heroku port variable for deployment)
CMD ["R", "-e", "shiny::runApp('/app', host = '0.0.0.0', port = as.numeric(Sys.getenv('PORT')))"]
```

在这个 docker 文件中，renv 用于通过“RUN Rscript -e 'renv::restore()'”从 renv.lock 文件安装必要的 R 库，以使容器中的库版本与本地开发环境中的库版本相同。

要在 Heroku 上运行容器化的应用程序，有两件事是必需的。首先，出于安全原因，容器必须以非 root 身份运行。通过`RUN useradd shiny_user`创建新用户，并通过`USER shiny_user`进行设置。

其次，Heroku 通过`PORT`主机变量为每个 dyno 提供一个随机端口。要在这个端口运行闪亮的应用程序，在 runApp 命令中使用`port = as.numeric(Sys.getenv('PORT'))`。

# 使用 GitHub 动作的持续部署

GitHub Actions 提供了一个简单的模板和[指令](https://github.com/marketplace/actions/deploy-to-heroku)从[https://github.com/AkhileshNS/heroku-deploy](https://github.com/AkhileshNS/heroku-deploy)部署到 Heroku。在顶层包括一个目录。GitHub repo 中包含以下 main.yml 文件的 github/workflows:

```
name: heroku_deploy
    on:                         
        push:                           
            branches:
                - masterjobs:                        
    build:                          
    runs-on: ubuntu-latest                        
       steps:
           - uses: actions/checkout@v2        
           - uses: akhileshns/heroku-deploy@v3.6.8
           with: 
               heroku_api_key: ${{secrets.HEROKU_API_KEY}}  
               heroku_app_name: "geocode-shiny"                                 
               heroku_email: ${{secrets.HEROKU_EMAIL}}                                   
               healthcheck: "https://geocode-shiny.herokuapp.com/"                                     
               usedocker: true                                  
               delay: 60                                 
               rollbackonhealthcheckfailed: true env: 
              HD_API_KEY: ${{secrets.MAPS_API_KEY}} # Docker env var
```

这里，我们指定在推送到主分支时发生动作“heroku_deploy”。在这些步骤中，触发动作的提交被检出以供工作流访问，在下一步中，它被推送到 Heroku，构建并部署。

需要参数 *heroku_api_key、heroku_app_name* 和 *heroku_email* 。去得到它们

1.  在网站上创建一个 Heroku 帐户。
2.  转到 Heroku 仪表盘或[下载 CLI 工具](https://devcenter.heroku.com/articles/heroku-cli)。
3.  在仪表盘中或通过`heroku create your_app_name`创建一个具有唯一名称的新应用。
4.  从您的 Heroku 帐户设置或通过`heroku auth:token`获取 API 密钥。
5.  将 Heroku 帐户的两个变量存储在 GitHub repo Settings->Secrets 中，命名为 HEROKU_API_KEY 和 HEROKU_EMAIL。

部署 docker 容器需要参数 *usedocker: true* 。此外，我们使用 url(在第一次成功部署后您就会知道)和 *healthcheck: true* 。通过*rollbackonhealthcheckfailed:true*，当应用程序的运行状况检查失败时，运行状况检查延迟 60 秒，延迟:60 秒，部署回滚到上一次提交。

这个闪亮的应用程序需要一个秘密的 API 密钥，外部 API 才能工作。它也保存为 GitHub secret，并作为 Docker 环境变量提供。要为您的应用程序设置 env 变量，请在它们前面加上 HD_。这在部署中被剥离，并且是区分构建和部署变量所必需的。不要把你的任何秘密直接放在 GitHub repos 的文件里！

就是这样！推送至主分支(或任何其他分支，您也可以使用标记来指定 main.yml 中的部署提交),并检查 GitHub Actions 选项卡是否部署成功。然后在仪表盘中或通过`heroku apps:info -a your_app_name`获取你的应用的网址。将此 url 添加到 main.yml 中的 healthcheck，以用于将来的部署版本。

# 为应用程序设置自定义 URL

要为您拥有/有权访问的 Heroku app 设置一个[自定义域，您需要:](https://devcenter.heroku.com/articles/custom-domains)

1.  用信用卡验证您的 Heroku 帐户。你不会发生任何费用的应用程序免费层，此外，需要选择付费服务。如果你验证，你会得到更多的动力和免费动力小时。
2.  使用`heroku domains:add [www.example.com](http://www.example.com) -a your_app_name`通过 dashboard 或 CLI 工具将您的域添加到应用程序。
3.  转到您的域名提供商，为 www.example.com 的[添加一个新的 CNAME 记录，指向您通过仪表板或`heroku domains -a your_app_name`获得的 Heroku DNS 目标。](http://www.example.com)
4.  通过`host [www.example.com](http://www.example.com)`检查 DNS 配置是否正确。