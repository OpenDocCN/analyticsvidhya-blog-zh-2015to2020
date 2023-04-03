# 汽车购买是有趣的雪花第 1 部分

> 原文：<https://medium.com/analytics-vidhya/car-buying-is-fun-with-snowflake-7decb8d7e774?source=collection_archive---------19----------------------->

和其他人一样，我发现买车过程既复杂又耗时。主要是因为一个人必须访问许多经销商的网站库存寻找完美的汽车与所有期望的选项。

看了几个经销商的网站后，我意识到他们看起来都差不多。我想知道用 Python 写一个 web scrapper 是否能帮助我一次性查询所有经销商的库存。

根据与过滤器的交互，网站似乎在使用 REST APIs 加载新数据。所以我打开浏览器的开发工具来搜索任何可以被 scrapper 利用的 REST 端点。

![](img/9a62258ac6823c666523f01a67bc6055.png)

表单数据下的端点

查看端点，可以清楚地看到，基于 WordPress 的经销商网站使用 VMS(我假设这是车辆管理系统的缩写)插件，该插件将所有数据保存在带有相应“CP”Id 的中央数据库中。通过在浏览器地址栏中随机更新端点 url 中的 cp 值，它获取了代表该特定经销商库存的 JSON 数据。

```
[https://vms.prod.convertus.rocks/api/filtering/?cp=1000&ln=en&pg=1&pc=15](https://vms.prod.convertus.rocks/api/filtering/?cp=1000&ln=en&pg=1&pc=15)
```

让我们使用 Notepad++ JSON viewer 插件来看看这个 JSON 数据，以了解它的结构。查看结果计数，很容易在查询字符串中与“pc”值建立关联。

![](img/17eefe2e01b9aaa08d4fda29093f6d26.png)

这里的每个记录代表一个特定的车辆数据，如 VIN、品牌、型号、价格等。

![](img/41b9307aa202f57bf970547d3b1ab203.png)

虽然在 API 中使用易于猜测和迭代的顺序数字 id 并不是一个好主意，但是在这种情况下，使用带有几行 Python 代码的循环和 JSON 文件中的转储 API 输出，可以很容易地从许多经销商那里获得数据。

```
import requestsfor x in range(1000,2700):
    url = "[https://vms.prod.convertus.rocks/api/filtering/?cp={0}&ln=en&pg=1&pc=5000](https://vms.prod.convertus.rocks/api/filtering/?cp={0}&ln=en&pg=1&pc=5000)".format(x)

    print(url) 
    try:
        resp = requests.get(url)
        f = open("vms_{0}.json".format(x), "w")
        f.write(resp.text)
    finally:
        f.close()
```

它为循环中的每个请求创建一个文件。

![](img/b20f46790378ab5675302f22f4fe491f.png)

在雪花中，让我们创建一个 DB、表、格式和内部命名的 Stage 来摄取 JSON 文件。

```
//Create database and a table to hold this row data
CREATE DATABASE VMS COMMENT = 'Vehicle Management System';
CREATE TABLE "VMS"."PUBLIC"."VMS_STAGE_1" ("JSON_DATA" VARIANT) COMMENT = 'Stage level 1 for VMS';//Create a format to ingest JSON VMS files
CREATE FILE FORMAT "VMS"."PUBLIC".vms_json_format TYPE = 'JSON' COMPRESSION = 'AUTO' ENABLE_OCTAL = FALSE ALLOW_DUPLICATE = FALSE STRIP_OUTER_ARRAY = FALSE STRIP_NULL_VALUES = FALSE IGNORE_UTF8_ERRORS = FALSE COMMENT = 'Format to ingest VMS JSON files';//Create an Internal Named Stage to upload data into Snowflake
CREATE STAGE "VMS"."PUBLIC".vms_stage COMMENT = 'Stage to upload VMS JSON files for ingestion';
```

下一步是使用 SnowSQL 工具上传 JSON 文件。打开 CMD / Terminal，导航到 JSON 文件下载的目录，启动 SNOWSQL。登录后:触发以下命令

```
//Upload all *.json files to stage
put file://*.json [@VMS_STAGE](http://twitter.com/VMS_STAGE);
```

这需要时间来执行，因为在上传到雪花阶段之前有许多文件需要压缩和加密

![](img/37e2637516ad19674456088f0d45099d.png)

将数据从上传文件复制到 1 级阶段表的时间

```
copy into vms.public.vms_stage_1 from [@vms_stage](http://twitter.com/vms_stage) file_format=(format_name=VMS_JSON_FORMAT);
```

花了大约 15 秒插入所有的数据，我认为它的速度很快，考虑到它是使用 XS 大小的仓库完成的。

![](img/92b755de4ce4ae2732ccb0cc28fd8295.png)

同样，是时候验证数据在阶段表中的外观了

![](img/ff3f96a44cd1a471790e8e0350ab13cc.png)

请记住，每一行都包含特定经销商所有车辆的数据。因此，要列出每辆车，需要对记录节点进行扁平化处理。

```
select value:vin::text as vin,
value:make::text as make, value:model::text as model, value:retail_price::double as retail_price
from vms.public.vms_stage_1
,LATERAL FLATTEN(input => json_data:results);
```

![](img/d083580cf18cf5ae6b8f4535c70109e1.png)

创建一个视图，将每辆车的“有趣”属性解析为一个单独的列

```
create or replace view vms_stage_2 as
select value:ad_id::text as id,value:stock_number::text as stock_number, value:vin::text as vin, value:days_on_lot::int as days_on_lot, value:sale_class::text as sale_class,value:demo::text as demo,
value:sale_expiry::text as sale_expiry, value:vehicle_class::text as vehicle_class, value:year::text as year,
value:make::text as make, value:model::text as model, value:trim::text as trim, value:passenger::text as passenger, 
value:retail_price::double as retail_price,value:lowest_price::double as lowest_price, value:asking_price::double as asking_price, value:internet_price::double as internet_price, value:final_price::double as final_price,
value:wholesale_price::double as wholesale_price, value:sales_tax::double as sales_tax,
value:odometer::int as odometer, value:fuel_type::text as fuel_type, value:transmission::text as transmission, value:engine::text as "ENGINE", value:drive_train::text as drive_train, value:doors::text as doors, 
value:exterior_color::text as exterior_color, value:vdp_url::text as vdp_url, value:image:image_original::text as image_original, value:manu_program::text as manu_program, 
value:manu_exterior_color::text as manu_exterior_color,
value:body_style::text as body_style, value:certified::int as is_certified, value:company_data:company_name::text as company_name,
value:company_data:company_city::text as company_city, value:company_data:company_province::text as company_province, value:company_data:company_sales_email::text as company_sales_email, 
value:company_data:company_sales_phone::text as company_sales_phone 
from vms.public.vms_stage_1
,LATERAL FLATTEN(input => json_data:results);
```

以便在执行该视图时，它返回最佳可用价格。

![](img/3e6bab8176280722749719681a90d842.png)

我喜欢用雪花语法解析 JSON 的简单性和它提供的性能。接下来，我将研究使用流来自动化这个工作流。

这是我第一次在媒体上发帖。希望你喜欢这篇文章。

[第 2 部分](https://paragshah.medium.com/car-buying-is-fun-with-snowflake-part-2-42a6b91d0870)发布。这一切都是为了自动化这个工作流程。