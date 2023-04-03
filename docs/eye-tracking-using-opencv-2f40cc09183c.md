# 使用 OpenCV 进行眼睛跟踪

> 原文：<https://medium.com/analytics-vidhya/eye-tracking-using-opencv-2f40cc09183c?source=collection_archive---------8----------------------->

在我 2000 年初的时候，我父亲经常开车带我去我的家乡度假。从我住的地方出发大概需要 8 个小时。在我们的旅程中，我的主要角色是坐在他身边，每当他的眼睛闭上时叫醒他。如果我不在他身边呢？是不是很恐怖？是啊。在阅读了 OpenCV 的这个案例研究后，我们可以使用一个小相机来跟踪你的眼睛，感到很高兴，并想象它将有助于避免高速公路上的许多事故。

# 追踪眼睛的步骤

1.  检测你的脸
2.  裁剪人脸区域并检测人脸区域中的眼睛
3.  继续追踪眼睛区域

# 代码片段

如果你在自己的视频中尝试，我们需要调整比例因子和最小邻居来获得更好的结果。

```
# import the necessary packages
import cv2class EyeTracker:
   def __init__(self, faceCascadePath, eyeCascadePath):
       # load the face and eye detector
       self.faceCascade = cv2.CascadeClassifier(faceCascadePath)
       self.eyeCascade = cv2.CascadeClassifier(eyeCascadePath) def track(self, image):
       # detect faces in the image and initialize the list of
       # rectangles containing the faces and eyes
       faceRects = self.faceCascade.detectMultiScale(image,
         scaleFactor = 1.1, minNeighbors = 5, minSize = (30,30),
         flags = cv2.CASCADE_SCALE_IMAGE)
       rects = [] # loop over the face bounding boxes
       for (fX, fY, fW, fH) in faceRects:
          # extract the face ROI and update the list of
          # bounding boxes
          faceROI = image[fY:fY + fH, fX:fX + fW]
          rects.append((fX, fY, fX + fW, fY + fH))

          # detect eyes in the face ROI
          eyeRects = self.eyeCascade.detectMultiScale(faceROI,
          scaleFactor = 1.1, minNeighbors = 10, minSize = (20, 20),
          flags = cv2.CASCADE_SCALE_IMAGE) # loop over the eye bounding boxes
          for (eX, eY, eW, eH) in eyeRects:
          # update the list of boounding boxes
          rects.append(
             (fX + eX, fY + eY, fX + eX + eW, fY + eY + eH)) # return the rectangles representing bounding
       # boxes around the faces and eyes
       return rects
```

所以现在我们创建了一个类，它可以从视频中检测人脸和眼睛。但是如果我们发送不同大小的视频，参数需要经常改变。为了克服这一点，我们可以对输入图像进行预处理，这样在转换成某种格式后将调用眼球跟踪器类。

```
# USAGE
# python eyetracking.py — face cascades/haarcascade_frontalface_default.xml — eye cascades/haarcascade_eye.xml — video video/adrian_eyes.mov
# python eyetracking.py — face cascades/haarcascade_frontalface_default.xml — eye cascades/haarcascade_eye.xml# import the necessary packages
from pyimagesearch.eyetracker import EyeTracker
from pyimagesearch import imutils
import argparse
import cv2# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(“-f”, “ — face”, required = True,
 help = “path to where the face cascade resides”)
ap.add_argument(“-e”, “ — eye”, required = True,
 help = “path to where the eye cascade resides”)
ap.add_argument(“-v”, “ — video”,
 help = “path to the (optional) video file”)
args = vars(ap.parse_args())# construct the eye tracker
et = EyeTracker(args[“face”], args[“eye”])# if a video path was not supplied, grab the reference
# to the gray
if not args.get(“video”, False):
   camera = cv2.VideoCapture(0)# otherwise, load the video
else:
   camera = cv2.VideoCapture(args[“video”])# keep looping
while True:
   # grab the current frame
   (grabbed, frame) = camera.read() # if we are viewing a video and we did not grab a
   # frame, then we have reached the end of the video
   if args.get(“video”) and not grabbed:
      break # resize the frame and convert it to grayscale
   frame = imutils.resize(frame, width = 300)
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # detect faces and eyes in the image
   rects = et.track(gray) # loop over the face bounding boxes and draw them
   for rect in rects:
      cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]),    (0, 255, 0), 2) # show the tracked eyes and face
   cv2.imshow(“Tracking”, frame) # if the ‘q’ key is pressed, stop the loop
   if cv2.waitKey(1) & 0xFF == ord(“q”):
     break# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows(
```

**参考:**

Adrian 的 OpenCV-基础包

链接代码:【https://github.com/RamjiB/Eye-Tracking 