# 用 Dialogflow 构建 Android 聊天机器人

> 原文：<https://medium.com/analytics-vidhya/build-android-chatbot-with-dialogflow-4a82a8809c6a?source=collection_archive---------0----------------------->

在本文中，我们将分享将 Google 的 Dialogflow 聊天机器人集成到您的 android 应用程序中的步骤。为 android 应用程序构建一个示例聊天机器人只需要 [Dialogflow](https://dialogflow.com/) 和[komunicate。](https://www.kommunicate.io/)

下面是一个使用 Dialogflow 在 android 中开发的**komunicate 支持机器人**的例子。我们实际上在我们的网站上使用这个机器人。如果你想看机器人的现场表演，点击这里。

可操作的富消息驱动的机器人可以根据用户是否在聊天中进行一般查询、技术查询或安排演示来回复。

[**创建免费聊天机器人**](https://dashboard.kommunicate.io/signup)

您可以使用现有的 dialogflow bot 或 checkout [bot samples](https://docs.kommunicate.io/docs/bot-samples) 来构建自己的合格 bot。从[这里](https://docs.kommunicate.io/samples/kommunicate-support-bot-sample.zip)下载 Kommunicate 支持机器人，并导入到您的 Dialogflow 帐户中。

![](img/eede6a2beedfd3da913286c56eb5163f.png)

# 第一步:在[通信](https://dashboard.kommunicate.io/signup)中设置一个账户

这相当简单。你可以在 Kommunicate 获得一个免费账户。注册并导航到[机器人部分](https://dashboard.kommunicate.io/bot)。点击对话框中的**设置**。

上传您的 Dialogflow 提供的客户端密钥。

# 第二步:将 Dialogflow 集成的 Android 聊天机器人 SDK 安装到您的应用程序中

## 将 Android SDK 添加到您的应用中

在你的 Android 应用中安装 Kommunicate 既简单又快捷。Kommunicate SDK 附带预配置的 dialogflow 集成。

[**免费试用 30 天**](https://dashboard.kommunicate.io/signup)
strong >安装
在您的 app build.gradle 依赖项中添加以下内容:

`implementation 'io.kommunicate.sdk:kommunicateui:2.1.0'`

**初始化 SDK**

`Kommunicate.init(context, APP_ID);`

您可以通过在 Kommunicate Dashboard 上注册来获得应用程序 ID。
如需完整文档，请参考此处的。

# 带有 Dialogflow 聊天机器人的示例 Android 应用程序

这是一个集成了 Dialogflow 聊天机器人的示例 android 应用程序。[下载](https://github.com/Kommunicate-io/Kommunicate-Android-Chat-SDK)并在 Android studio 中运行。

# 第三步:使用 Dialogflow 集成聊天机器人启动 Android 聊天应用程序

现在，您可以通过聊天屏幕向 Dialogflow 发送有效负载，并从 Dialogflow 代理获得文本响应。
komunicate 提供了一个随时可用的聊天用户界面，因此这一步只需要启动聊天屏幕。

以下是使用 Dialogflow bot 启动聊天的说明:

```
new KmConversationBuilder(activityContext)
       .launchConversation(new KmCallback() {
           [@Override](http://twitter.com/Override)
           public void onSuccess(Object message) {
               Log.d("Conversation", "Success : " + message);
           }[@Override](http://twitter.com/Override)
           public void onFailure(Object error) {
               Log.d("Conversation", "Failure : " + error);
           }
       });
```

仅此而已！运行 Android 应用程序并与 Dialogflow bot 聊天。您可以通过几个简单的步骤轻松地将 Dialogflow 集成到 Android 应用程序中。如果您需要更多信息，您可以查看[通信文档。](https://docs.kommunicate.io/docs/android-installation)

# Dialogflow Android SDK

Dialogflow 提供了 Android 客户端 SDK，可以轻松地在 Android 设备上集成语音识别和 API.ai 自然语言处理 API。如果您希望开发自己的聊天实现，并在不使用 Kommunicate SDK 的情况下将 Dialogflow 直接集成到您的 Android 应用程序中，请集成以下 SDK。
[Github—dialog flow Android 客户端 SDK](https://github.com/dialogflow/dialogflow-android-client)

[**免费试用**](https://dashboard.kommunicate.io/signup)

*原载于 2018 年 11 月 15 日*[*https://www . komunicate . io*](https://www.kommunicate.io/blog/build-chatbot-with-dialogflow-android-sdk/)*。*