# 在 SparkR 中设置隔离的虚拟环境

> 原文：<https://medium.com/analytics-vidhya/setting-up-isolated-virtual-environments-in-sparkr-ff4795db8cfc?source=collection_archive---------13----------------------->

![](img/09903775dbd987cf79b2c92b159e0bd8.png)

# 动机

随着 Spark 越来越多地被用于扩展 ML 管道，如果我们想使用 UDF，安装和部署我们自己的 R 库变得尤为重要。在我的[上一篇文章](https://shbhmrzd.medium.com/stl-and-holt-from-r-to-sparkr-1815bacfe1cc)中，我谈到了使用 SparkR UDFs 在 R 中缩放 ML 管道。
今天，我将讨论为我们的 SparkR 运行设置一个虚拟环境，确保运行时依赖项和库安装在集群上。

# 限制

对于任何 Spark 集群，我们都可以在集群中的所有节点上安装 R 和所需的库，或者根据需要创建虚拟环境。
在我的例子中，我们有一个 Cloudbreak 集群，它只能访问边缘节点来提交 Spark 作业。所有其他群集节点都不可访问。
由于这些限制，我无法在边缘节点或集群上安装 R 和任何依赖项。

# 创造环境

因为我们目前在 R 中运行 ML 算法，所以我们有一个带有 R 的 docker 映像，所有的 ML 库都安装在它上面。我创建了一个新的映像，在其上安装了 Spark (v2.3.0，与 Cloudbreak cluster 相同)。

在这个容器上成功执行 ML 算法的 SparkR 实现[使用较小的数据集]确保了我可以使用这个 R 安装目录在 CloudBreak 集群上设置虚拟环境。由于权限限制，我们不能直接在 Cloudbreak 集群上安装 R，所以我打算将 R 安装目录从容器转移到边缘节点。

**install_spark.sh** :安装 spark 的 Shell 脚本。

```
yum -y install wgetwget — no-check-certificate [https://www.scala-lang.org/files/archive/scala-2.11.8.tgz](https://www.scala-lang.org/files/archive/scala-2.11.8.tgz)tar xvf scala-2.11.8.tgz
mv scala-2.11.8 /usr/lib
ln -sf /usr/lib/scala-2.11.8 /usr/lib/scala
export PATH=$PATH:/usr/lib/scala/binwget — no-check-certificate [https://archive.apache.org/dist/spark/spark-2.3.0/spark-2.3.0-bin-hadoop2.7.tgz](https://archive.apache.org/dist/spark/spark-2.3.0/spark-2.3.0-bin-hadoop2.7.tgz)tar xvf spark-2.3.0-bin-hadoop2.7.tgz
mkdir /usr/local/spark
cp -r spark-2.3.0-bin-hadoop2.7/* /usr/local/sparkexport SPARK_EXAMPLES_JAR=/usr/local/spark/examples/jars/spark-examples_2.11–2.3.0.jarln -sf /usr/bin/python3 /usr/bin/python
export PATH=$PATH:/usr/local/spark/bin#Installation directory of R on container : /usr/lib64/R 
```

**Dockerfile** : Dockerfile，用于基于安装了 R 和 ML 库的现有镜像创建安装了 Spark 的新镜像。

```
FROM <image_with_R_and_ML_libs_installed>:latest
COPY install_spark.sh ./
RUN bash install_spark.sh
ENV SPARK_EXAMPLES_JAR=”/usr/local/spark/examples/jars/spark-examples_2.11–2.3.0.jar”
```

# 引导环境

## 火花在本地模式下运行

我在 edge 节点主目录中创建了一个文件夹 *sparkr_packages* ，并将 R 安装目录和包从容器中复制到这里。

我们还需要设置一些必需的环境变量。

```
export PATH=$HOME/sparkr_packages/R/bin:$PATH
export R_LIBS=$HOME/sparkr_packages/R/library
export RHOME=$HOME/sparkr_packages/R
export R_HOME=$HOME/sparkr_packages/R
```

R 安装需要某些编译时依赖项，安装后就不需要了。因为我们已经成功地在容器上安装了 R 并通过了验证，所以我们不需要边缘节点上的这些依赖项。

我们仍然需要在 RScript 执行期间所需的运行时依赖项。如果没有这些库，启动 R 控制台将会失败，并显示如下错误

> $ HOME/sparkr _ packages/R/bin/exec/R:加载共享库时出错:libtre.so.5:无法打开共享对象文件:没有这样的文件或目录

在我的例子中，我需要边缘节点上的 *libtre.so.5* 和*libpcre 2–8 . so . 0*。

这些库也存在于容器中的 */usr/lib64/* 。就像 R 安装目录一样，我也将它们复制到位于 *sparkr_packages* 的边缘节点。
我们需要设置 LD_LIBRARY_PATH 指向这个位置，以便 R 运行时访问这些库。我们还可以将这些库添加到 R/libs 中，使它们在 R 运行时可用。

```
export LD_LIBRARY_PATH=$HOME/sparkr_packages:$LD_LIBRARY_PATH
```

我们现在可以在本地模式下启动 SparkR 控制台，并运行 UDF 来验证边缘节点上的安装。

## 火花发生器以集群模式运行

对于在集群模式下运行的 SparkR 使用 UDF，R 安装目录和运行时依赖项必须存在于所有执行器上。我们还需要在每个执行器上设置相应的环境变量。

我们可以使用 spark-submit 运行时参数*存档*将压缩的 sparkr _ packages 目录发送给所有的执行器。

> - archive:接受一个逗号分隔的归档列表，该列表将被提取到每个执行器的工作目录中。

在 spark-submit 过程中，我们可以通过使用 config*spark . executorenv .<property _ name>=<property _ value>*为每个执行器设置 R_HOME、LD_LIBRARY_PATH、PATH 等环境变量。

最后，启动 SparkR 会话

```
sparkR —-master yarn —-conf spark.executorEnv.RHOME=./environment/sparkr_packages/R —-conf spark.executorEnv.R_HOME_DIR=./environment/sparkr_packages/R —-conf spark.executorEnv.PATH=./environment/sparkr_packages/R/bin:$PATH —-conf spark.executorEnv.LD_LIBRARY_PATH=./environment/sparkr_packages:$LD_LIBRARY_PATH —-num-executors 10 —-executor-cores 3 —-executor-memory 10g —-archives sparkr_packages.zip#environment
```

# 结论

像这样设置虚拟环境有点麻烦，因为我们必须手动维护 R 可执行文件和模块。
尽管如此，这种方法对我们非常有用，让我们能够在不访问集群节点的情况下建立虚拟环境。