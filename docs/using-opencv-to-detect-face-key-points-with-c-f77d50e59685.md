# 用 OpenCV 在 C++中检测人脸关键点

> 原文：<https://medium.com/analytics-vidhya/using-opencv-to-detect-face-key-points-with-c-f77d50e59685?source=collection_archive---------20----------------------->

这篇文章是我的第一篇关于用 C++中的 OpenCV 构建人脸检测器的文章的后续。在这篇文章中，我们将建立在现有的代码和检测面部关键点。结果会是这样的。

![](img/840e1695166ceeb6c0eb7717e078c3a8.png)

由于我们将使用 OpenCV 的一个相对较新的版本(4.2.0 ),你可能想回到[以前的文章](https://bewagner.github.io/programming/2020/04/12/building-a-face-detector-with-opencv-in-cpp/)去阅读更多关于如何安装必要的包。

代码[在我的 github 上。](https://github.com/bewagner/visuals/tree/blog-post-2)

我们走吧！

# 检测人脸关键点

在上一篇文章中检测人脸后，我们现在想检测人脸关键点。我们使用`[cv::face::FacemarkLBF](https://docs.opencv.org/3.4/dc/d63/classcv_1_1face_1_1FacemarkLBF.html)`模型来寻找我们在上一课中确定的面部矩形中的关键点。

# 添加关键点检测模型文件

至于人脸检测模型，我们要为 LBF 模型添加一个[模型文件](https://github.com/bewagner/visuals/blob/blog-post-2/assets/lbfmodel.yaml)。我把模型文件放在[的`assets`文件夹里，这是 git repo](https://github.com/bewagner/visuals/tree/blog-post-2) 的帖子，你可以去那里下载。

为了将这个模型文件的位置传递给我们的代码，我们将使用与第一篇文章中相同的 CMake `target_compile_definitions`技巧。因此，确保您将模型文件放在了正确的位置，并将下面几行添加到您的`CMakeLists.txt`中。

```
# Introduce preprocessor variables to keep paths of asset files 
... 
set(KEY_POINT_DETECTION_MODEL 
  "${PROJECT_SOURCE_DIR}/assets/lbfmodel.yaml") 
... 
target_compile_definitions(${PROJECT_NAME} 
  PRIVATE KEY_POINT_DETECTION_MODEL="${KEY_POINT_DETECTION_MODEL}")
```

# 关键点检测器的类

我们首先为关键点检测器添加一个类。这样我们就有了在一个地方初始化和调用模型的代码。

## `KeyPointDetector.h`

在`include`文件夹中，我们创建一个文件`KeyPointDetector.h`。这将是关键点检测器的头文件。

`KeyPointDetector`将有两个公共方法。第一个是构造函数。我们将使用构造函数来初始化底层的 LBF 模型。

第二种方法`detect_key_points`检测图像中给定矩形内的人脸关键点。由于每个面的关键点都属于类型`std::vector<cv::Point2f>`，这个函数将返回一个矢量`std::vector<cv::Point2f>`。

关键点检测器的头文件如下所示。

```
#ifndef KEYPOINTDETECTOR_H 
#define KEYPOINTDETECTOR_H  
#include <opencv4/opencv2/face.hpp>
class KeyPointDetector { 
public:
     /// Constructor
     explicit KeyPointDetector();
      /// Detect face key points within a rectangle inside an image     
      /// \param face_rectangles Rectangles that contain faces
     /// \param image Image in which we want to detect key points
     /// \return List of face keypoints for each face rectangle
     std::vector<std::vector<cv::Point2f>>
     detect_key_points(const std::vector<cv::Rect> &face_rectangles,
                       const cv::Mat &image) const;
  private:
     cv::Ptr<cv::face::Facemark> facemark_;
 };
#endif //KEYPOINTDETECTOR_H
```

## `KeyPointDetector.cpp`

接下来，我们在`src/KeyPointDetector.cpp`中实现这些方法。

首先，我们来看看构造函数。我们创建了一个新的`cv::face::FacemarkLBF`模型。然后，我们从通过 CMake 传入的`KEY_POINT_DETECTION_MODEL`变量加载模型配置。

```
KeyPointDetector::KeyPointDetector() {
     facemark_ = cv::face::FacemarkLBF::create();
     facemark_->loadModel(KEY_POINT_DETECTION_MODEL);
 }
```

下面，我们实现`detect_key_points`。

为了遵守`cv::face::Facemark::fit()`的 API，我们将输入转换为`cv::InputArray`。然后我们调用 models `fit`函数，返回检测到的点。

```
std::vector<std::vector<cv::Point2f>>
 KeyPointDetector::detect_key_points(
         const std::vector<cv::Rect> &face_rectangles,
          const cv::Mat &image) const
  { cv::InputArray faces_as_input_array(face_rectangles);
     std::vector<std::vector<cv::Point2f> > key_points;
     facemark_->fit(image,
              faces_as_input_array,
              key_points);
      return key_points;
  }
```

# 使用关键点检测器

现在我们跳到`main.cpp`来使用我们定义的关键点检测器。我们使用[上一篇文章](https://bewagner.github.io/programming/2020/04/12/building-a-face-detector-with-opencv-in-cpp/)中的人脸检测器。然后，我们将检测到的矩形提供给我们的关键点检测器。

```
#include <opencv4/opencv2/opencv.hpp> 
#include "FaceDetector.h" 
#include "KeyPointDetector.h"  
int main(int argc, char **argv) {
      cv::VideoCapture video_capture;
     if (!video_capture.open(0)) {
         return 0;
     } FaceDetector face_detector;
     KeyPointDetector keypoint_detector; cv::Mat frame;
     while (true) {
         video_capture >> frame;
         auto rectangles = face_detector
                 .detect_face_rectangles(frame);
          auto keypoint_faces = keypoint_detector
                 .detect_key_points(rectangles, frame);
```

我们显示检测到的点，而不是显示矩形。

```
 const auto red = cv::Scalar(0, 0, 255);
        for (const auto &face :keypoint_faces) {
            for (const cv::Point2f &keypoint : face) {
                cv::circle(frame, keypoint,
                           8, red, -1);
            }
        }

        imshow("Image", frame);
        const int esc_keycode = 27;
        if (cv::waitKey(10) == esc_keycode) {
            break;
        }
    }
    cv::destroyAllWindows();
    video_capture.release();
    return 0;
}
```

您应该会看到类似下图的结果。

![](img/840e1695166ceeb6c0eb7717e078c3a8.png)

# 结论

在这篇文章中，我们使用了人脸检测模型来寻找图像中的人脸。然后我们使用 OpenCV 在这些图像中找到关键点。

我希望这能帮助你建立有趣的东西！这里有一个[代码](https://github.com/bewagner/visuals/tree/blog-post-2)的链接。如果你遇到任何错误，请告诉我！

在 twitter 上关注我 [(@bewagner_)](https://twitter.com/bewagner_) 了解更多关于 C++和机器学习的内容！

*本帖原载于*[*https://be Wagner . github . io/programming/2020/04/23/detecting-face-key points-with-opencv/*](https://bewagner.github.io/programming/2020/04/23/detecting-face-keypoints-with-opencv/)