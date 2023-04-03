# Swift 和 Firebase——数据源复制？

> 原文：<https://medium.com/analytics-vidhya/swift-and-firebase-datasource-repeating-dea13af8e367?source=collection_archive---------11----------------------->

## **充满自信地聆听和观察的隔音方法。**

![](img/ca10cff9f4c371f18316b626a36db77d.png)

# P 先决条件

通过谷歌(实时数据库)对 Swift 和 Firebase 的基本了解。

## 这是干什么用的？

从实时数据库中加载任何类型的 JSON 数据，无论是加载社交提要还是下拉用户列表。这里的关键要素是以 100%的准确度进行观察和自动更新。此外，当数据完成加载时，我们将调用一个完成处理程序。我们开始吧！

## **跳跃中**

在所有选择后端的选项中，你选择了 Google 的 Firebase，实时数据库。明白了。当你开始在你的 UICollection 和 UITableViews 中填充优秀的内容时，砰的一声，重复的悲剧发生了。您正在删除侦听器并清除数据源数组。有什么问题！？让我们用一种完全证明的方法来解决这个问题，无所畏惧地观察和倾听。

## **观察概述**

```
Step 1: .observe(.valueStep 2: .observeSingleEvent(of: .valueStep 3: .observe(.childAdded
```

## 履行

让我们创建一个新类，并在 viewDidLoad 函数上方添加一些变量，然后导入 Firebase。

```
let databaseRef = Database.database().reference()var observingRefOne = Database.database().reference()var handleOne = DatabaseHandle()var compileCounter : Int = 0
```

## 功能

让我们创建一个名为 fetchCurrentChatMessages 的函数，它在完成时进行转义。

```
func fetchCurrentChatMessages(completion : @escaping (_ isComplete : Bool)->()) {}
```

步骤 1:让我们检查授权并添加观察者。这将监听路径末端的任何变化，并再次触发该函数。确保在这里使用自己的路径。

```
func fetchCurrentChatMessages(completion : @escaping (_ isComplete : Bool)->()) {guard let user_uid = Auth.auth().currentUser?.uid else {return}let ref = self.databaseRef.child(“messages”).child(user_uid)ref.observe(.value, with: { (snapListener : DataSnapshot) in}) { (error) incompletion(false)}}
```

第 2 步:让我们添加 compileCounter 变量，看看我们要遍历多少个对象，并确保数据确实存在。如果没有，我们稍后将调用失败状态。

```
func fetchCurrentChatMessages(completion : @escaping (_ isComplete : Bool)->()) {guard let user_uid = Auth.auth().currentUser?.uid else {return}let ref = self.databaseRef.child(“messages”).child(user_uid)ref.observe(.value, with: { (snapListener : DataSnapshot) inref.observeSingleEvent(of: .value, with: { (snapCount : DataSnapshot) inif snapCount.exists() {let snapChildrenCount = snapCount.childrenCount} else if !snapCount.exists() {completion(false)}}) { (error) incompletion(false)}}) { (error) incompletion(false)}}
```

第三步:现在是棘手的部分。让我们添加一个观察器，它遍历所有现有对象并连接我们的句柄，同时将我们的 compileCounter 变量增加 1。

```
func fetchCurrentChatMessages(completion : @escaping (_ isComplete : Bool)->()) {guard let user_uid = Auth.auth().currentUser?.uid else {return}let ref = self.databaseRef.child(“messages”).child(user_uid)ref.observe(.value, with: { (snapListener : DataSnapshot) inref.observeSingleEvent(of: .value, with: { (snapCount : DataSnapshot) inif snapCount.exists() {let snapChildrenCount = snapCount.childrenCountself.observingRefOne = self.databaseRef.child(“messages”).child(user_uid)self.handleOne = self.observingRefOne.observe(.childAdded, with: { (snapLoop : DataSnapshot) inself.compileCounter += 1guard let json = snapLoop.value as? [String : Any] else {return}//APPEND YOUR OBJECT DATA AND ORGANIZE/FILTER AS NEEDEDif self.compileCounter == snapChildrenCount {completion(true)}}, withCancel: { (error) incompletion(false)})} else if !snapCount.exists() {completion(false)}}) { (error) incompletion(false)}}) { (error) incompletion(false)}}
```

一旦我们的 compileCounter 变量与我们的 snapChildrenCount 常量匹配，我们就调用完成处理程序，因为我们的数据已经被加载了！现在我们需要移除我们的观察者。我们的完成处理程序应该根据它的完成状态调用两个函数。首先，确保在 viewDidLoad 中调用 fetchCurrentChatMessages 函数，或者将它放入一个新函数中，以保持代码的整洁。

```
self.fetchCurrentChatMessages { (isComplete) inif isComplete == true {self.handleSuccess()} else if isComplete == false {self.handleFailure()}}
```

现在，添加 handleSuccess 和 handleFailure 函数。

```
func handleSuccess() {self.compileCounter = 0self.observingRefOne.removeObserver(withHandle: self.handleOne)DispatchQueue.main.async {//RELOAD YOUR COLLECTION OR TABLE VIEW}}func handleFailure() {self.compileCounter = 0self.messageArray.removeAll() // REPLACE THIS WITH YOUR DATASOURCEself.observingRefOne.removeObserver(withHandle: self.handleOne)DispatchQueue.main.async {//RELOAD YOUR COLLECTION OR TABLE VIEW}}
```

最后，确保在再次调用 observer 后，清除 datasource 数组并重置 compileCounter 变量。最终的 fetchCurrentChatMessages 函数应该如下所示。

```
func fetchCurrentChatMessages(completion : @escaping (_ isComplete : Bool)->()) {guard let user_uid = Auth.auth().currentUser?.uid else {return}let ref = self.databaseRef.child(“messages”).child(user_uid)ref.observe(.value, with: { (snapListener : DataSnapshot) inself.compileCounter = 0self.messageArray.removeAll() // REPLACE THIS WITH YOUR DATASOURCEref.observeSingleEvent(of: .value, with: { (snapCount : DataSnapshot) inif snapCount.exists() {let snapChildrenCount = snapCount.childrenCountself.observingRefOne = self.databaseRef.child(“messages”).child(user_uid)self.handleOne = self.observingRefOne.observe(.childAdded, with: { (snapLoop : DataSnapshot) inself.compileCounter += 1guard let json = snapLoop.value as? [String : Any] else {return}//APPEND YOUR OBJECT DATA AND ORGANIZE/FILTER AS NEEDEDif self.compileCounter == snapChildrenCount {completion(true)}}, withCancel: { (error) incompletion(false)})} else if !snapCount.exists() {completion(false)}}) { (error) incompletion(false)}}) { (error) incompletion(false)}}
```

就是这样！您现在可以监听/观察并自动更新您的 UICollection 和 UITableViews，而无需重复数据！请根据需要随意复制和粘贴！