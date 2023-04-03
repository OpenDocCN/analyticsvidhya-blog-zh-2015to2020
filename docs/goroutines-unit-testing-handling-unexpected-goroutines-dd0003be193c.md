# Goroutines 单元测试(处理 goroutines 泄漏)

> 原文：<https://medium.com/analytics-vidhya/goroutines-unit-testing-handling-unexpected-goroutines-dd0003be193c?source=collection_archive---------6----------------------->

我将解释我们如何编写 golang 单元测试，并在测试完成后发现意外的错误。

假设我们有这样一个场景，作为一个客户端，我们想要发出一个 http 请求，它在内部创建多个 goroutines 并异步发出多个 http 调用来获取数据并返回给客户端累积的 http 响应。如果任何 http 请求失败或返回错误响应，那么客户端需要得到一个错误响应。在这种情况下，如果任何 http 请求失败，那么我们不想等待其他 http 请求完成。一旦我得到一个错误，我需要发送错误响应给客户端。

下面是获取给定城市列表的天气详细信息的代码片段。

```
**package** main

**import** (
   **"errors"
   "fmt"
   "io/ioutil"
   "net/http"** )

**type** WeatherClient **struct** {
   client  *http.Client
   baseURL string
}

**func** NewWeatherClient(client *http.Client, baseURL string) *WeatherClient {
   **return** &WeatherClient{
      client:  client,
      baseURL: baseURL,
   }
}

**func** (wc WeatherClient) FindWeatherForCities(cities []string) []string {
   weatherData := make([]string, 0)
   responseChan := make(**chan** string, 1)
   errorChan := make(**chan** error, 1)

   **for** city := **range** cities {
      cityWeatherURL := fmt.Sprintf(wc.baseURL+**"?q=%s&appid=weatherAPIApplicationId"**, city)
      **go** wc.GetWeatherForCity(cityWeatherURL, responseChan, errorChan)
   }

   **for** {
      **select** {
      **case** <-errorChan:
         **return** weatherData[:0]
      **case** resp := <-responseChan:
         weatherData = append(weatherData, resp)
      **default**:
      }

      **if** len(weatherData) == len(cities) {
         **return** weatherData
      }
   }
}

**func** (wc WeatherClient) GetWeatherForCity(url string, responseChan **chan** string, errorChan **chan** error) {
   **defer func**() {
      **if** r := recover(); r != nil {
         fmt.Println(**"RECOVER"**, r)
         errorChan <- errors.New(**"error panic goroutine"**)
      }
   }()

   resp, err := wc.client.Get(url) **if** resp.StatusCode == 200 {
      bytes, err := ioutil.ReadAll(resp.Body)
      **if** err != nil {
         errorChan <- err
         **return** }
      responseChan <- string(bytes)
   } else {
      errorChan <- err
   }
}
```

使用 base URL“https://[api.openweathermap.org/data/2.5/weather](http://samples.openweathermap.org/data/2.5/weather?q=London,uk&appid=b6907d289e10d714a6e88b30761fae22)”和“默认 http 客户端”创建一个新的 WearherClient。

将“**weatherAPIApplicationId”**替换为正确的应用 Id，以获得城市列表的响应。

现在，如果无法获得任何城市的数据，它将向客户端发送错误响应，因为我们不想显示部分数据。

**测试上面的代码**。

```
**package** main

**import** (
   **"github.com/stretchr/testify/assert"
   "go.uber.org/goleak"
   "net/http"
   "net/http/httptest"
   "strings"
   "testing"** )

**func** TestPrintWeatherForCities(t *testing.T) {
   **defer** goleak.VerifyNone(t)
   cities := []string{**"city1"**,**"city2"**, **"city3"**, **"city4"**, **"city5"**, **"city6"**, **"city7"**, **"city8"**}

   server := httptest.NewServer(http.HandlerFunc(**func**(w http.ResponseWriter, r *http.Request) {
      **if** strings.EqualFold(**"q=%!s(int=0)&appid=weatherAPIApplicationId"**, r.URL.RawQuery) {
         w.WriteHeader(500)
         **return** }
      w.WriteHeader(200)
      w.Write([]byte(**"Passed"**))
   }))

   baseURL := server.URL
   weatherClient := NewWeatherClient(server.Client(), baseURL)
   weatherData := weatherClient.FindWeatherForCities(cities)
   server.Close()

   assert.Equal(t, []string{}, weatherData)
}
```

因为我正在使用由 **uber** 开发的名为 **goleak"** 的包来捕获上述代码中的意外 goroutines，所以测试将失败，并显示错误:“**发现意外 go routines**”。并且如果注释行" **defer goleak。验证无(t)** “测试将通过。但是这不是使它通过测试的正确方法，相反，应该使用适当的编程逻辑来修复它。

**怎么修？**

在代码片段中，我创建了两个大小为 1 的有界通道来收集来自 goroutines 的响应和错误。

因此，如果任何 goroutine 产生错误，上述程序将退出，但多个 go routine 仍将运行，这就是 goleak verify 将捕获的 go routine。

处理这个问题的一种方法是创建具有给定城市列表大小的频道。这将有助于处理运行中的 goroutines 响应/错误。

所以只要随着城市的长度改变响应的大小和误差通道的大小。

```
responseChan := make(**chan** string, len(cities))
errorChan := make(**chan** error, len(cities))
```

可能有其他可能的解决方案来处理正在运行的 goroutines，直到它正确地完成进程。

想要解释“**goleak**的用法”并避免在所有 goroutine 完成之前通道关闭时出现恐慌情况。

欢迎评论/反馈/建议。

谢了。