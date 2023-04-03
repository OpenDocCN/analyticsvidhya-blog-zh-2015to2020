# 在 IntelliJ 上用 Maven 测试 Hadoop 程序

> 原文：<https://medium.com/analytics-vidhya/testing-your-hadoop-program-with-maven-on-intellij-42d534db7974?source=collection_archive---------1----------------------->

在本教程中，我们将了解如何在 IntelliJ 上使用 Maven 编写和测试 Hadoop 程序，而无需在自己的机器上配置 Hadoop 环境或使用任何集群。这不是一篇[字数统计 map reduce 代码教程](https://dzone.com/articles/6-free-vue-react-and-spring-ebooks-learn-full-stac)假设对 map-reduce 功能有基本的理解。

## 要求

*   软件开发工具包(Software Development Kit)
*   [IntelliJ](https://www.jetbrains.com/idea/download/#section=mac) (点击下载)
*   Linux 或 Mac OS

## *创建新项目*

单击创建新项目并选择 Maven，然后单击下一步

![](img/d7195df8da4968967ffe5e717d8202bc.png)![](img/f8e21837587ab3a015ea619e6d9fab77.png)

设置项目名称、项目位置、groupId 和 artifactId。保持该版本不变，然后单击 finish。

![](img/b137a310c6f743e328bb60b6cf49d327.png)

*现在我们已经准备好配置我们的项目依赖关系了*

# 配置依赖关系

打开 pom.xml 文件。该文件通常是单击“完成”后的默认打开屏幕。单击“启用自动导入”,但如果您希望每次编辑 pom.xml 文件时都收到通知，也可以导入更改。

![](img/b6fa29756cbcd5ceb7d9b58ab9d059c4.png)

在您的 pom.xml 文件中，在项目结束标记之前发布以下代码块

```
<repositories>
    <repository>
        <id>apache</id>
        <url>http://maven.apache.org</url>
    </repository>
</repositories><dependencies>
    <dependency>
        <groupId>org.apache.hadoop</groupId>
        <artifactId>hadoop-core</artifactId>
        <version>1.2.1</version>
    </dependency>
    <dependency>
        <groupId>org.apache.hadoop</groupId>
        <artifactId>hadoop-common</artifactId>
        <version>3.2.0</version>
    </dependency>
</dependencies>
```

最终的 pom 文件应该如下所示

![](img/031c7ac0444e2481e187291a0092aacb.png)

下面是完整的 pom.xml 文件

```
<?xml version="1.0" encoding="UTF-8"?>
<project 
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>org.Word</groupId>
    <artifactId>WordCount</artifactId>
    <version>1.0-SNAPSHOT</version>
    <repositories>
        <repository>
            <id>apache</id>
            <url>http://maven.apache.org</url>
        </repository>
    </repositories>
    <dependencies>
        <dependency>
            <groupId>org.apache.hadoop</groupId>
            <artifactId>hadoop-core</artifactId>
            <version>1.2.1</version>
        </dependency>
        <dependency>
            <groupId>org.apache.hadoop</groupId>
            <artifactId>hadoop-common</artifactId>
            <version>3.2.0</version>
        </dependency>
    </dependencies>

</project>
```

*现在我们准备为我们的示例测试项目 WordCount 创建类。*

## 创建 WordCount 类

进入 src -> main -> java package 并创建一个新类

![](img/7ddd1dbe991d6b77db40552efdad5cb6.png)

命名该类，然后单击并输入

![](img/e26360fd3c26979491784d88482a1e24.png)

将以下 Java 代码粘贴到 wordCount 类中。

```
import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class wordCount {

    public static class TokenizerMapper
            extends Mapper<Object, Text, Text, IntWritable> {

        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            StringTokenizer itr = new StringTokenizer(value.toString());
            while (itr.hasMoreTokens()) {
                word.set(itr.nextToken());
                context.write(word, one);
            }
        }
    }

    public static class IntSumReducer
            extends Reducer<Text, IntWritable, Text, IntWritable> {
        private IntWritable result = new IntWritable();

        public void reduce(Text key, Iterable<IntWritable> values,
                           Context context
        ) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "word count");
        job.setJarByClass(wordCount.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setCombinerClass(IntSumReducer.class);
        job.setReducerClass(IntSumReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
```

![](img/0cdbfa94500c589344d903c2bf58de4d.png)![](img/81d106d8181bc3cba23e9d47b9fbefae.png)

wordCount 类代码包括 main 方法、map 类和 reduce 类。它扫描由第一个参数定义的文件夹中的所有文本文件，并将所有单词的频率输出到由第二个参数定义的文件夹中。

## 我们几乎准备好运行程序了…

首先，我们必须创建文本输入文件。在您的项目包中创建新的文件夹，并将其命名为 input。然后在输入文件夹/目录中创建你的 txt 文件，或者如果你已经有了的话拖动一个。

![](img/ffc9b9a6a8197945db356bdc2be7ea38.png)![](img/6a5bf273c403f3865c8eac8398b98a07.png)![](img/b4f10f291b5dd076d8340644e75364b7.png)![](img/c8e98b5d2f01f68dd63b7b8530f05efa.png)![](img/5aa67a401bd061a5d7c07c894581c773.png)

在这个文件中复制并粘贴一些文本

*快好了，耐心点……*

我们还没有设置我们的程序参数。选择运行→编辑配置。

![](img/0ba6928fb97a7df2038106d1d1366375.png)

通过选择“+”然后选择应用程序来添加新的应用程序配置。

![](img/a40e59063f78317504d1a50841f4ac59.png)

设置`Main class`为字数，设置`Program arguments`为输入输出。这允许程序从输入文件夹读取并将结果保存到`output`文件夹。不要创建输出文件夹，因为 Hadoop 会自动创建该文件夹。如果该文件夹存在，Hadoop 将引发一个异常。完成后，选择应用，然后选择确定。

![](img/33486711d092a59b0b18514468d999bd.png)

现在我们准备好运行我们的程序了。

选择`Run` → `Run 'WordCount'`运行 Hadoop 程序。如果重新运行程序，请删除之前的输出文件夹。

![](img/fea14ac0a90577fe04436ae9cf649b4c.png)

将出现一个输出文件夹。每次运行的结果都保存在*输出* → *part-r-00000* 中。

![](img/f664c51826ccef892ddde68af7ba1932.png)

# Mac 上可能出现的问题

如果您的 mac 上运行的是最新版本的 Java，您可能会遇到以下错误。

![](img/a488441e67603f85ea1fc42a3693ab91.png)

## 解决办法

我的系统使用 Javac 版本 9 来编译程序，所以我将以下内容设置为我的 Javac 编译器版本。

文件->项目结构->项目->项目 SDK -> 9。
文件- >项目结构- >项目- >项目语言级别- > 9。
文件- >项目结构- >项目- >模块- > - >源代码→ 9…
项目中- > ctrl + alt + s - >构建、执行、部署- >编译器- > Java 编译器- >项目字节码版本- > 9。
Intellij IDEA - >构建、执行、部署- >编译器- > Java 编译器- >模块- > 1.9。