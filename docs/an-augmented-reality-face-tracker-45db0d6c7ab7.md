# 一种增强现实人脸追踪器

> 原文：<https://medium.com/analytics-vidhya/an-augmented-reality-face-tracker-45db0d6c7ab7?source=collection_archive---------16----------------------->

我在寻找增强现实教程，我到处都是。我仍然认为最好的选择是 Unity，我喜欢使用它。但是这篇文章更多的是让你开始。我会告诉你基于网络的增强现实和虚拟现实。

如果你想自己动手，我会在这里为你列出一些 **资源**:

1.  [https://medium . com/@ urish/web-powered-augmented-reality-a-hands-on-hands-tutorial-9e6a 882 e 323 e](/@urish/web-powered-augmented-reality-a-hands-on-tutorial-9e6a882e323e)
2.  [https://medium . com/@ yupingohanga/start-coding-augmented-reality-bdd3c 546595](/@yupingohanga/start-coding-augmented-reality-bdd3c546595)
3.  [https://medium . com/@ fauzali/creating-web-based-augmented-reality-with-just-10-lines-of-html-code-for-初学者-ar-js-d62ef596eab](/@fauziali/creating-web-based-augmented-reality-with-just-10-lines-of-html-code-for-beginners-ar-js-d62ef596eab)
4.  达尼洛·帕斯夸列洛讲座([https://www.youtube.com/watch?v=ktjMCanKNLk&list = pl 8 mkb hej 75 fjd-hvedzm 4 xkrcic 5 VF YUV](https://www.youtube.com/watch?v=ktjMCanKNLk&list=PL8MkBHej75fJD-HveDzm4xKrciC5VfYuV))

但是，如果你想继续，请继续。最终的结果会是这样的:[**https://jojo96.github.io/Ar-face-tracker/**](https://jojo96.github.io/Ar-face-tracker/)

你可以从[**https://github.com/jojo96/Ar-face-tracker**](https://github.com/jojo96/Ar-face-tracker)获得所需文件

让我们开始:

我们从 index.html 的档案开始。

```
<!DOCTYPE html>
<html><head>
    <script src="[https://aframe.io/releases/1.0.3/aframe.min.js](https://aframe.io/releases/1.0.3/aframe.min.js)"></script>
    <script src="[https://libs.zappar.com/zappar-aframe/0.2.4/zappar-aframe.js](https://libs.zappar.com/zappar-aframe/0.2.4/zappar-aframe.js)"></script><meta charset="utf-8" />
    <title>Zappar for A-Frame: Face Tracking 3D Model Example</title>
    <style>
        html,
        body {
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
        }
    </style>
</head>
```

这两行非常重要:

```
**<script src="**[**https://aframe.io/releases/1.0.3/aframe.min.js**](https://aframe.io/releases/1.0.3/aframe.min.js)**"></script>
    <script src="**[**https://libs.zappar.com/zappar-aframe/0.2.4/zappar-aframe.js**](https://libs.zappar.com/zappar-aframe/0.2.4/zappar-aframe.js)**"></script>**
```

他们安装 Zapworks 提供的所需 javascript 库。该代码采用并修改自 **Zapworks github 账户:**([https://github.com/zappar-xr/zappar-aframe-examples](https://github.com/zappar-xr/zappar-aframe-examples))

剩下的代码也非常简单。

**启动前置摄像头:**

<a-scene></a-scene>

**启动面部追踪器:**

**使用**加载投射到面部的模型

<a-entity gltf-model="”url(<strong" class="ih hj">assets/z _ helmet . glb)" position = " 0.3-1.3 0 " scale = " 1.1 1.1 1.1 "></a 实体></a-entity>

这里的 **z_helmet.glb(3d 模型)**位于资产文件夹中。您可以根据需要更改路径。如果你从我的 github 下载文件，这应该没有任何变化。

```
<body>
    <a-scene>
        <a-camera zappar-camera="user-facing: true;" /><!-- Setup our face tracker -->
        <a-entity zappar-face id="my-face-tracker"><!-- Include a head mask object that will make sure the user's head appears in the center of our helmet -->
            <a-entity zappar-head-mask="face:#my-face-tracker;"></a-entity>

            <!-- Include a 3D model inside our face tracker -->
             <a-entity gltf-model="url(assets/z_helmet.glb)" position="0.3 -1.3 0" scale="1.1 1.1 1.1"></a-entity></a-entity>
    </a-scene>
</body></html>
```

这样，编码就完成了。

恭喜你！你已经准备好面部追踪器了。

**整个代码组合在一起:**

```
<!DOCTYPE html>
<html><head>
    <script src="[https://aframe.io/releases/1.0.3/aframe.min.js](https://aframe.io/releases/1.0.3/aframe.min.js)"></script>
    <script src="[https://libs.zappar.com/zappar-aframe/0.2.4/zappar-aframe.js](https://libs.zappar.com/zappar-aframe/0.2.4/zappar-aframe.js)"></script><meta charset="utf-8" />
    <title>Zappar Example</title>
    <style>
        html,
        body {
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
        }
    </style>
</head><body>
    <a-scene>
        <a-camera zappar-camera="user-facing: true;" /><!-- Setup our face tracker -->
        <a-entity zappar-face id="my-face-tracker"><!-- Include a head mask object that will make sure the user's head appears in the center of our helmet -->
            <a-entity zappar-head-mask="face:#my-face-tracker;"></a-entity>

            <!-- Include a 3D model inside our face tracker -->
             <a-entity gltf-model="url(assets/z_helmet.glb)" position="0.3 -1.3 0" scale="1.1 1.1 1.1"></a-entity></a-entity>
    </a-scene>
</body></html>
```

要运行此程序:

1.  从:([https://www.npmjs.com/package/@zappar/zapworks-cli](https://www.npmjs.com/package/@zappar/zapworks-cli))下载 zapworks cli
2.  转到命令行(从下载的文件夹)，然后键入:`zapworks serve .`
3.  转到 [https://127.0.0.1:8080](https://127.0.0.1:8080/)

**进一步的项目**

下面是一个您可以使用自己的原语的示例:

```
<!DOCTYPE html>
<html><head>
    <script src="[https://aframe.io/releases/1.0.3/aframe.min.js](https://aframe.io/releases/1.0.3/aframe.min.js)"></script>
    <script src="[https://libs.zappar.com/zappar-aframe/0.2.4/zappar-aframe.js](https://libs.zappar.com/zappar-aframe/0.2.4/zappar-aframe.js)"></script><meta charset="utf-8" />
    <title>Own Primitives Example</title>
    <style>
        html,
        body {
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
        }
    </style>
</head><body>
    <a-scene>
        <a-camera zappar-camera="user-facing: true;" /><!-- Setup our face tracker -->
        <a-entity zappar-face id="my-face-tracker"><!-- Include a head mask object that will make sure the user's head appears in the center of our helmet -->
            <a-entity zappar-head-mask="face:#my-face-tracker;"></a-entity>

   <a-circle
   color="#FFC107"
   side="double"
   position = "0 0 0"
   scale = "1 1 1"
   >
   </a-circle>

   <a-box 
   color = "#FFC107"
   position ="0 1 0"
   scale = "1 1 1">
   </a-box>

</a-entity>

    </a-scene>
</body></html>
```

参考文献:
1。[https://github.com/zappar-xr/zappar-aframe-examples](https://github.com/zappar-xr/zappar-aframe-examples)

2.[https://zap.works/universal-ar/aframe/](https://zap.works/universal-ar/aframe/)