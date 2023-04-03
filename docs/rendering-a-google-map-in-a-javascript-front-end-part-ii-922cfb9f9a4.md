# 在 JavaScript 前端呈现谷歌地图

> 原文：<https://medium.com/analytics-vidhya/rendering-a-google-map-in-a-javascript-front-end-part-ii-922cfb9f9a4?source=collection_archive---------12----------------------->

## 第二部分:存储和渲染地图上的点和多边形

![](img/ea490a51031a82f694d010ab2b4decf7.png)

在我的[上一篇文章](/@lukemenard/rendering-a-google-map-in-a-javascript-front-end-part-i-a076420a9ee9)中，我讨论了我在谷歌云平台上建立一个项目并通过谷歌地图 JavaScript API 在 JavaScript 前端呈现地图的方法。在本文中，我将详细介绍在渲染地图中生成新的点和多边形数据的方法，这些数据乐观地和悲观地保存在 Ruby-on-Rails 后端数据库中。

我创建的这个项目是一个网络工具，用来帮助消防员和护林员追踪野火和计划规定的烧伤。出于我的目的，我想让最终用户能够绘制可编辑、可删除和持久的新的点和面数据。

# 后端

## 创建 Rails API

为了创建持久的 Google Maps 点和多边形数据，我首先构建了一个 rails 后端数据库和 API 来存放信息，然后通过 API 在前端检索必要的数据。我创建了一个“点”模型，其中包括用于标题、描述、链接的字符串/文本字段，以及用于点的纬度和经度的浮动字段:

```
create_table "points", force: :cascade *do* |t|
   t.string "title"
   t.float "latitude"
   t.float "longitude"
   t.string "link"
   t.text "description"
   t.datetime "created_at", null: false
   t.datetime "updated_at", null: false
*end*
```

我还用相同的字符串和文本字段生成了一个“多边形”模型。然而，我为潜在多边形顶点的纬度和经度包括了 15 个不同的浮点字段。这是一个有点随意的选择，将应用程序中创建的新多边形限制为 14 条边。

```
create_table "polygons", force: :cascade *do* |t|
   t.string "title"
   t.string "link"
   t.string "description"
   t.float "lat1"
   t.float "long1"
   t.float "lat2"
   t.float "long2"
   t.float "lat3"
   t.float "long3" ... t.float "lat15"
   t.float "long15"
   t.datetime "created_at", null: false
   t.datetime "updated_at", null: false
*end*
```

此外，我为点和多边形模型创建了控制器文件，包括相关的 CRUD 方法，并将生成的 JSON 数据呈现给 Rails API。

```
*class* PointsController < ApplicationController
   *def* index
      points = Point.all
      render json: points, except: [:updated_at, :created_at]
   *end**def* create
      point = Point.create(title: params[:title], latitude:
      params[:latitude], longitude: params[:longitude], link:
      params[:link], description: params[:description])
      render json: point, except: [:updated_at, :created_at]
   *end**def* show
      point = Point.find_by(id: params[:id])
      render json: point, except: [:updated_at, :created_at]
   *end**def* update
      point = Point.find_by(id: params[:id])
      point.update(point_params)
      render json: point, except: [:updated_at, :created_at]
   *end**def* destroy
      point = Point.find_by(id: params[:id])
      point.delete
   *end*private*def* point_params
      params.require(:point).permit(:id, :title, :latitude,
      :longitude, :description, :link)
   *end
end*
```

出于我的目的，运行本地 Rails 服务器进行前端数据检索就足够了。

# 前端

## 创建新点数据

当用户以预定的方式与地图交互时，应该创建新的数据点。Google Maps JavaScript API 有几个内置的[事件监听器](https://developers.google.com/maps/documentation/javascript/events)(例如，点击、双击、鼠标悬停)，它们可以在服务器端完成所选动作时触发一种方法。我构建了一个事件侦听器，它通过创建一个新的数据点来响应地图显示上的鼠标双击。

```
google.maps.event.addListener(map, 'dblclick', function(event){
})
```

在事件监听器中，我创建了一个新的 Google Maps 标记，其位置设置为点击事件的纬度和经度。

```
let marker = new google.maps.Marker({
   position: event.latLng,
   map: map,
   draggable: true
})
```

或者，您可以创建一个与创建时打开的每个新点相关联的新信息窗口。我还创建了一个空白的“div”元素作为 info 窗口中的占位符内容。

```
let div = document.createElement('div')let infoWindow = new google.maps.InfoWindow({
   content: div
})
infoWindow.open(map, marker)
```

在事件监听器中，我还创建了一个表单，当存储在后端数据库中时，允许用户输入与每个新点相关联的内容。该表单包括一系列输入字段来收集相关的点信息，存储为 innerHTML。

```
let form = document.createElement('form')
form.innerHTML =
` <div class='new-Form'>
   <form>
      <h3> Create New Point </h3>
      <input type="hidden" id="id" name="id"  value="id">
      Title: <input type="text" id="title" name="title"  value="">
      Description: <inputtype="textContent" id="description" name="description"  value="">
      Latitude: <input id="latitude" name="latitude"  value="${event.latLng.lat()}">
      Longitude: <input id="longitude" name="longitude"  value="${event.latLng.lng()}">
      Link: <input type="text" id="link" name="link"  value="">
   </form>
</div> `
```

此外，我创建了一个带有 click 事件的表单提交按钮，单击该按钮将触发后续的“getFormData”函数(稍后讨论)。然后，我将 form 和 submit 按钮添加到每个新 info 窗口中包含的“div”元素中。

```
let submitButton = document.createElement('button')
submitButton.innerText = 'Create New Point'
submitButton.addEventListener('click', function(event){
   event.preventDefault()
   getFormData()
})div.append(form, submitButton)
```

双击并创建一个新的点，一个信息窗口完成上述表格和提交按钮应该打开，并允许用户输入。

在提交 info window 表单时，我需要收集用户输入的数据，以便在后面的方法中使用。我通过调用一个“getFormData”函数来实现这一点，该函数利用 JavaScript FormData 构造函数来检索每个用户输入并将其存储到一个变量中。

```
function getFormData(){
   let formData = new FormData(form)
   let title = formData.get('title')
   let description = formData.get('description')
   let latitude = formData.get('latitude')
   let longitude = formData.get('longitude')
   let link = formData.get('link')addNewPoint(title, description, latitude, longitude, link)
}
```

“getFormData”调用一个附加函数“addNewPoint”，该函数接收用户输入数据，触发对已建立的 API 的 HTTP POST 请求，并向后端数据库添加一个新条目。

```
function addNewPoint(title, description, latitude, longitude, link){
   marker.setMap(null)let clickconfig = {
      method: "POST",
      headers: {
         "Content-Type": "application/json",
         "Accept": "application/json"
      },
      body: JSON.stringify({
         title,
         description,
         latitude,
         longitude,
         link
      })
   }fetch(POINTS_URL, clickconfig)
      .then(response => response.json())
      .then(renderPointCard)
      .catch(function(error){
         console.log(error.message)
      })
   }
})
```

该函数将序列化的用户数据提交给 API，并将结果解析为 JSON。然后，我将结果数据传递给一个额外的函数“renderPointCard”，该函数在新点数据上创建额外的样式和单击功能，但这超出了本文的范围。

刷新页面时，新创建的点应该保留在页面上。

## 创建新的多边形数据

谷歌地图 JavaScript API 有一个内置的特性，[绘图管理器](https://developers.google.com/maps/documentation/javascript/drawinglayer)，允许用户轻松绘制新的多边形(或圆形、矩形等)。)在地图上。我在“initMap”函数中创建了一个新的绘图管理器实例，将多边形选择器按钮的位置设置为地图的底部中心，并定制了多边形表示的样式。

```
let drawingManager = new google.maps.drawing.DrawingManager({
   drawingMode: null,
   drawingControl: true,
   drawingControlOptions: {
      position: google.maps.ControlPosition.BOTTOM_CENTER,
      drawingModes: ['polygon']
   },
   polygonOptions: {
      fillColor: '#ff5733',
      fillOpacity: 0.3,
      strokeWeight: 1,
      clickable: true,
      editable: true,
      zIndex: 1,
   }
});
drawingManager.setMap(map)
```

如上所述，我创建了一个新的数据点，类似地，我创建了一个地图级别的事件监听器，它由绘图管理器多边形的完成来触发。

```
google.maps.event.addListener(drawingManager, 'polygoncomplete', function(polygon){
}
```

完成后，多边形的数据通过一个更大的函数传递，该函数标识每个多边形顶点的纬度和经度。我通过使用 Google Maps API 的内置“getPaths()”方法来识别组成多边形的所有标签的对象，并将结果值存储到一个变量中，从而实现了这一点。面要素是一个嵌套对象，需要一些钻孔来识别各个纬度和经度(请注意。g 下面的链条)。使用 for 循环，我将各个纬度和经度分开，并存储在一个新的数组中。

```
let p = polygon.getPaths("latLngs").g[0].glet latLng = []
*for*(let i = 0; i < p.length; i++){
   let lat = 0
   lat = p[i].lat()

   let long = 0
   long = p[i].lng()latLng.push(lat)
   latLng.push(long)
}
```

我按照上面详述的相同过程为每个新多边形生成一个信息窗口。每个 info 窗口都包含一个可编辑的表单和 submit 按钮，该按钮触发一个方法来检索和存储用户输入到表单中的数据。

```
let div = document.createElement('div')let form = document.createElement('form')
form.innerHTML =
`<div>
   <form>
      <h3> Create New Polygon </h3>
      <input type="hidden" id="id" name="id"  value="id">
      <input type="hidden" id="polygon_id" name="polygon_id"  value="polygon_id">
      <input type="hidden" id="lat1" name="lat1" value="${latLng[0]}"
      <input type="hidden" id="lat2" name="lat2" value="${latLng[2]}">...<input type="hidden" id="lat15" name="lat15" value="${latLng[28]}"
      <input type="hidden" id="long1" name="long1" value="${latLng[1]}">
      <input type="hidden" id="long2" name="long2" value="${latLng[3]}">...

      <input type="hidden" id="long15" name="long15" value="${latLng[29]}">Title: <input type="text" id="title" name="title"  value="">
      Description: <input type="textContent" id="description" name="description"  value="">
      Link: <input type="text" id="link" name="link"  value=""></form>
</div>`let polySubmitButton = document.createElement('button')
polySubmitButton.innerText = 'Create New Wildfire'
polySubmitButton.addEventListener('click', function(event){
   event.preventDefault()
   getPolygonFormData()
})div.append(form, polySubmitButton)
```

在这种情况下，每个多边形顶点的纬度和经度值通过隐藏的表单字段传递给“getPolygonFormData”函数。

```
function getPolygonFormData(){
   let formData = new FormData(form)
   let title = formData.get('title')
   let description = formData.get('description')
   let link = formData.get('link')
   let lat1 = formData.get('lat1')
   let long1 = formData.get('long1')
   let lat2 = formData.get('lat2')
   let long2 = formData.get('long2')...let lat15 = formData.get('lat15')
   let long15 = formData.get('long15')}
```

然后，该函数将表单数据传递给后续函数“addNewPolygon”，后者执行对 API 的 HTTP POST 请求，并向后端数据库添加一个新的多边形条目。

```
function addNewPolygon(
   title,
   description,
   link,
   lat1,
   long1,
   lat2,
   long2,...

   lat15,
   long15
   ){let polyConfig = {
      method: "POST",
      headers: {
         "Content-Type": "application/json",
         "Accept": "application/json"
      },
      body: JSON.stringify({
         title,
         description,
         link,
         lat1,
         long1,
         lat2,
         long2,...

         lat15,
         long15
      })
   }
   fetch(POLYGONS_URL, polyConfig)
   .then(response => response.json())
   .then(renderPolygonCard)
   .catch(function(error){
      console.log(error.message)
   })
}})
```

与上面描述的点数据方法一样，这个函数将序列化的用户数据提交给 API，并将结果承诺解析为 JSON。然后，我将结果数据传递给一个附加函数“renderPolygonCard”，该函数在新点多边形数据上创建附加的样式和单击功能。

刷新页面时，新创建的多边形也应该保留在页面上。

有关我如何使用普通 JavaScript 在点和多边形数据上启用完整 CRUD 方法的更多详细信息，请随意访问我的完整 GitHub 存储库，在这里可以找到:

[](https://github.com/lukemenard/Wildfire-Tracker) [## Luke menard/野火追踪器

### 一个基于谷歌地图的 Javascript 网络应用程序，使用野火追踪器近乎实时地追踪美国的野火…

github.com](https://github.com/lukemenard/Wildfire-Tracker) 

随时欢迎投稿和评论！