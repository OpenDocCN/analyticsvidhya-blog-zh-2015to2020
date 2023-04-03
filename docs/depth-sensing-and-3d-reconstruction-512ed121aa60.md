# 用于三维重建的立体视觉

> 原文：<https://medium.com/analytics-vidhya/depth-sensing-and-3d-reconstruction-512ed121aa60?source=collection_archive---------1----------------------->

今年早些时候，特斯拉分享了一段令人印象深刻的视频，展示了他们的 3D 传感技术。他们的系统接收相机图像并输出周围的点云，这几乎可以与激光雷达点云相媲美。

这有什么了不起的？嗯，当我们在 2D 拍摄照片时，由于一个称为 [*透视投影*](https://en.wikipedia.org/wiki/3D_projection#Perspective_projection) 的过程，所有的深度信息都丢失了，获得 3D 点云对于自动驾驶技术至关重要，因为它可以精确测量道路上的各种物体。深度感测是计算机视觉中的一项流行任务，并且存在各种方法来解决该问题。端到端学习方法越来越受欢迎，其中带注释的深度图可以被神经网络作为输入，并被训练来为任何输入图像生成深度图。但是也有基于基本原则的传统方法。因此，我不想直接跳到深度学习，而是想看看这种方法的表现。从图像中获取深度信息是可能的，我们甚至可以获得点云，但我们不能只用一张图像来做到这一点。这种方法被称为立体视觉，它的工作原理与人眼非常相似。通过比较左右眼看到的东西，我们可以了解环境的深度。有很多资源可以更好地理解这项技术，所以让我们继续。

我发现通过实践学习非常有效，所以让我们使用[道路/车道检测](http://www.cvlibs.net/datasets/kitti/eval_road.php)的 KITTI 基准数据来快速浏览一下这种方法。

![](img/ef234b51a4315b759556842b49c5c7b8.png)

从这组立体图像中

![](img/7f8d8ce01b3bc05ee9b03c0d2f1082ce.png)

我们获得一个三维点云

## 立体视觉三维点云重建

第一步是加载左右图像，并从立体图像中获取视差图。立体图像组的视差图像被定义为这样的图像，其中每个像素表示图像一中的像素与其在图像二中匹配的像素的*之间的距离。有几种方法可以做到这一点，我们将使用 opencv StereoBM 中提供的块匹配方法。我们将通过为块匹配算法设置 numDisparities、blockSize 来初始化对象，以获得良好的视差图，compute()的输入是灰度的左右图像。虽然有更好的算法来计算视差，但对于这个演示来说已经足够好了。OpenCV 提供了[其他匹配算法](https://docs.opencv.org/3.4/dd/d86/group__stereo.html)，可能值得一试，除此之外还有很多文献。注意，我们可以直接进行视差图的生成，因为我们已经校正了图像，但是在其他情况下，首先不失真图像是很重要的。*

```
stereo = cv2.StereoBM_create(numDisparities=96, blockSize=11)
disparity = stereo.compute(img_left_gray,img_right_gray)
```

![](img/f868d100095f750da6ff820a6e3d434d.png)

视差图

正如我们从上面的图像中看到的，匹配并不总是完美的，在差异中有一些噪声和漏洞。此外，左栏是空白的，因为视差在左图像的帧中，而在右图像中没有对应的白色补丁。因此，当我们重建点云时，我们将会丢失这部分，对称地，我们也丢失了右边图像的右栏。

现在我们有了视差图，我们可以用它来获得带有 XYZRGB 信息的 3D 点云。为此，我们将需要转换视差图，以便我们可以获得深度信息，为此我们需要视差-深度矩阵，让我们通过 [stereoRectify()](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#stereorectify) 来计算。我们可以提供两幅图像的校准信息作为输入，它将输出视差-深度投影矩阵。由于两幅图像已经是未失真的并且在校正的坐标系中，我们将只需要两个相机矩阵和它们之间的平移 T。该矩阵 T 由 [KITTI 传感器设置](http://www.cvlibs.net/datasets/kitti/setup.php)计算得出。

```
cam1 = calib_matrix_P2[:,:3] # left color image
cam2 = calib_matrix_P3[:,:3] # right color imagerev_proj_matrix = np.zeros((4,4)) # to store the outputcv2.stereoRectify(cameraMatrix1 = cam1,cameraMatrix2 = cam2,
                  distCoeffs1 = 0, distCoeffs2 = 0,
                  imageSize = img_left_color.shape[:2],
                  R = np.identity(3), T = np.array([0.54, 0., 0.]),
                  R1 = None, R2 = None,
                  P1 =  None, P2 =  None, 
                  Q = rev_proj_matrix)
```

现在我们可以使用 [reprojectImageTo3D()](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#reprojectimageto3d) 将视差图投影到 3D 点云上。

```
points = cv2.reprojectImageTo3D(img, rev_proj_matrix)
```

我们会注意到，点云是镜像的，这是意料之中的，因为图像中的 X 轴是镜像的，我们可以使用简单的 2D 反射沿 X 轴来解决这个问题。

让我们存储获得的点和颜色信息，我们将重用 opencv [3D 重建示例](https://github.com/opencv/opencv/blob/master/samples/python/stereo_match.py)中的一些代码。要从图像中提取颜色添加到点云，我们只需从图像中提取颜色，并对其进行整形以匹配点云形状。

```
# opencv loads the image in BGR, so lets make it RGB
colors = cv2.cvtColor(img_left_color, cv2.COLOR_BGR2RGB)
# resize to match point cloud shape
colors = colors.resize(-1, 3)
```

我们现在可以将 XYZRGB 点云写入 ply 文件，这是一种常用的点云格式，可以用 [Meshlab](http://www.meshlab.net/) 可视化。

```
write_ply('out.ply', out_points, out_colors)
```

看着点云非常有趣，因为它非常好地捕捉了深度信息，让人很难相信它是从一对 2D 图像中获得的，请在 meshlab 中从各个角度随意欣赏点云。即使在下图中，一些缺陷也很明显，但在大多数情况下，更接近地面的表面看起来很好，有了更好的视差图，就更容易提高点云质量。

![](img/a7eb67fdced10b4eef870868d0c02d95.png)

在 meshlab 中可视化，使用点装饰着色

## 投影到图像

既然我们已经倒退了，为了获得原始的(某种)点云，我们可以再次前进到我们开始的图像。将三维点云投影到图像帧上有许多应用。此外，这是一个很好的方法来验证点云的质量，以及我们遵循的过程是完整和无损的。

通常，该步骤需要点云和图像坐标系之间的外部校准，因为点云是基于图像帧生成的，它们之间没有旋转和平移，并且因为图像已经未失真，也没有失真系数。请记住，我们翻转了 X 轴，所以我们将通过应用相同的反射将其翻转回来。然后使用 opencv [projectPoints()](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#projectpoints) 使用相机矩阵来投影这些点。该函数将返回每个 3D 点的 2D 投影点。

```
cv2.projectPoints(reflected_points, np.identity(3), 
                  np.array([0., 0., 0.]),
                  cam1_matrix, np.array([0., 0., 0., 0.]))
```

最后，我们可以通过将每个像素绘制为圆形()来将点绘制到图像中，此时我们将使用前面的颜色数组，其索引与 2D 投影点对齐。

```
for i, pt in enumerate(projected_points):
    pt_x = int(pt[0])
    pt_y = int(pt[1])
    # use the BGR format to match the original image type
    col = (colors[i, 2], colors[i, 1], colors[i, 0])
    cv2.circle(blank_img, (pt_x, pt_y), 1, col)
```

![](img/c97bc6e377cd4f006aea50d75a3dd620.png)

我的 github 上有一个自动处理 Kitti 数据集的 [python 笔记本](https://github.com/umangkshah/notebooks/tree/master/3d_reconstruction)。请随意看一看，亲自尝试一下。对于上述主题的详细解释，您可以通过整篇文章中的链接来获得。感谢阅读。