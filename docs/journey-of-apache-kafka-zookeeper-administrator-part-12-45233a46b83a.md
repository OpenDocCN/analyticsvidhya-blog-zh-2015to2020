# 阿帕奇·卡夫卡与动物园管理员之旅(12)

> 原文：<https://medium.com/analytics-vidhya/journey-of-apache-kafka-zookeeper-administrator-part-12-45233a46b83a?source=collection_archive---------20----------------------->

2020 年 6 月(Oracle 云上的 Terraform Magic)

正如我在上一篇关于 AWS 上的 Terraform 的文章中解释的那样。

![](img/016c7b72aea127926b002db2f833babc.png)

T 今天，我将使用 Terraform 解释 Oracle 云设置。
这将是非常类似的 AWS 设置只是名称术语的变化夫妇。

**测试代码:**[116 dav inder/Kafka-cluster-ansi ble/terraform/OCI](https://github.com/116davinder/kafka-cluster-ansible/tree/master/terraform/oci)

**Terraform** 需要很多东西才能开始 aka [**var.tf**](https://github.com/116davinder/kafka-cluster-ansible/blob/master/terraform/oci/var.tf)

*   **根租户 ID**:ocid 1 . tenancy . xxxxxxxxxxxxxxxxxxxxxxx zwtxryat 3 zbha
*   **用户 ID:**ocid 1 . User . xxxxxxxxxxxxxxxxxxxxxxxxxxxx XV 5 shh ccn 7 uglfoxpsa
*   **隔离舱 ID:**ocid 1 . Compartment . xxxxxxxxxxxxxxxxxxx NW 3 yxaackla
*   **用户 ID 指纹:**F7:16:FD:xxxxxxxxxxxxxxxxxxxxxxxxxxx:ab:D4:ee:9a
*   **用户私有 PEM 文件:**/home/davenderpal/。oci/oci_api_key.pem
*   **用户地区:**欧盟-法兰克福-1
*   **甲骨文 Linux ID *:**ocid 1 . image . oc1 . eu-Frankfurt-1 . xxxxmmawia 2 womz 5 q
*   **虚拟云网络 ID**:ocid 1 . vcn . oc1 . eu-Frankfurt-1 . xxxxxx75 cua
*   **子网 ID ***:ocid 1 . Subnet . oc1 . eu-Frankfurt-1 . xxxxxxxxxxxxapzt 3 vzuc 2 q
*   **实例 ID:** 虚拟机。标准 2.2
*   **AZ ID *:**ExfZ:EU-法兰克福 1-AD-1
*   **安全列表 ID *:**ocid 1 . Security List . oc1 . eu-Frankfurt-1 . xxxxxxxx 33 giwiva
*   **SSH 公钥:**SSH-RSA xxxxxxxxxxxxxxxxxxxxx RSA-Key 2047 @ da vinder _ Pal

**注*:** 1。星号(*)变量可以自动化但我没来得及这么做。
2。当前设置与 HA 不兼容，因为使用了单个 AZ。
3。我已经为每个变量添加了关于如何/从哪里在 Oracle Cloud 中收集它们的描述，因此请仔细阅读[**var . TF**](https://github.com/116davinder/kafka-cluster-ansible/blob/master/terraform/oci/var.tf)**位。**

> **开始行动吧！**

**我们需要 **EBS** 又名**块存储**在 **Oracle Cloud** 中。**

```
resource "oci_core_volume" "kafka-volume" {
    count = var.kafka_instance_count
    availability_domain = var.instance_availability_domain
    compartment_id = var.compartment_ocid
    display_name = "kafka-volume-${count.index + 1}"
    size_in_gbs = var.kafka_block_volume_size_gb
}
```

**我们需要创建实例。**

```
resource "oci_core_instance" "kafka" {
    count = var.kafka_instance_count
    availability_domain = var.instance_availability_domain
    compartment_id = var.compartment_ocid
    shape = var.instance_shapeagent_config {
        is_monitoring_disabled = false
    }
    create_vnic_details {
        subnet_id = var.subnet_ocid
        assign_public_ip = var.use_public_ip
    }
    metadata = {
        ssh_authorized_keys = var.ssh_public_key
    }
    display_name = "kafka-${count.index + 1}"
    source_details {
        source_id = var.instance_image_ocid
        source_type = "image"
    }
    preserve_boot_volume = false
}
```

**最后，我们需要将块存储与实例关联起来。(使用 **iSCSI****

```
resource "oci_core_volume_attachment" "kafka_volume_attachment" {
    count = var.kafka_instance_count
    attachment_type = "iscsi"
    instance_id = element(oci_core_instance.kafka.*.id, count.index)
    volume_id = element(oci_core_volume.kafka-volume.*.id, count.index)display_name = "kafka-volume-attachment-${count.index + 1}"
    is_pv_encryption_in_transit_enabled = false
    is_read_only = false
    use_chap = false
}
```

****最后，如何从 Linux 连接到卷？****

**附加卷后，您可以配置 iSCSI 连接。使用`iscsiadm`命令行工具连接到卷。控制台提供了配置、验证和登录所需的命令，因此您可以轻松地将它们复制并粘贴到实例会话窗口中。配置连接后，您可以在实例上挂载卷，并像使用物理硬盘驱动器一样使用它。**

**若要连接到您的宗卷:**

1.  **按照[连接到您的实例](https://docs.cloud.oracle.com/en-us/iaas/Content/GSG/Tasks/testingconnection.htm#Connecting_to_Your_Instance)中所述，登录到您的实例。**
2.  **打开导航菜单。在**核心基础设施**下，转到**计算**并点击**实例**。**
3.  **单击实例名称以查看其详细信息。**
4.  **在**资源**部分，点击**附加块卷**。**
5.  **点击你刚刚连接的卷旁边的操作图标(三个点),然后点击 **iSCSI 命令和信息**。**
6.  **显示 **iSCSI 命令和信息**对话框。请注意，该对话框显示了有关您的宗卷的特定标识信息(如 IP 地址和端口)以及您需要使用的 iSCSI 命令。这些命令已经准备好使用，并且每个命令中已经包含了适当的信息。**
7.  ****附加命令**配置 iSCSI 连接并登录到 iSCSI。将**附加命令**列表中的每个命令复制并粘贴到实例会话窗口中。**
8.  **请确保分别粘贴和运行每个命令。有三个附加命令。每个命令都以`sudo iscsiadm`开头。**
9.  **输入登录 iSCSI 的最后一个命令后，您就可以格式化(如果需要)和挂载卷了。要获取实例上可装载的 iSCSI 设备的列表，请运行以下命令:**

*   **`sudo fdisk -l`**

**如果您的磁盘连接成功，您将在返回的列表中看到它，如下所示:**

*   **`Disk /dev/sdb: 50.0 GB, 50010783744 bytes, 97677312 sectors Units = sectors of 1 * 512 = 512 bytes Sector size (logical/physical): 512 bytes / 512 bytes I/O size (minimum/optimal): 4096 bytes / 1048576 bytes`**

**请使用以下链接获取最新信息。**

**[https://docs.cloud.oracle.com/en-us/Tasks/addingstorage.htm](https://docs.cloud.oracle.com/en-us/iaas/Content/GSG/Tasks/addingstorage.htm)**

****注意:** Terraform 需要相当长的时间，因为 Oracle 云的性能不可预测，有时需要 5-6 分钟，有时需要 10-15 分钟。**

**现在，Ansible 可以用于 Kafka 软件的实际设置。你可以重温我以前的文章。**

**这个旅程将在下一篇文章中继续(Kafka 2.6.0 和 Java 11 的变化)**