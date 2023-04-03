# 阿帕奇卡夫卡与动物园管理员之旅(十一)

> 原文：<https://medium.com/analytics-vidhya/journey-of-apache-kafka-zookeeper-administrator-part-11-197d4ea232d0?source=collection_archive---------11----------------------->

2020 年 6 月(AWS 云上的 Terraform Magic)

![](img/3d8f039b8950518bf0ad25843a4af172.png)

礼貌:数据狗

如果你正在阅读我的文章，那么你会明白，到目前为止，我还没有谈到在云上运行卡夫卡。这里面有起有落。

让我们以#AWS 云为例。

**AWS** 提供 **MSK** 为卡夫卡提供**托管服务。它解决了 Kafka 的许多问题，集群设置可以在几分钟内完成，而不是手动几个小时。对于 MSK，我们也失去了控制，所以这是一个权衡，你需要决定你是应该学习一切还是仅仅依靠 AWS 的支持。**

我建议您应该在 AWS 中拥有一个没有 MSK 的手动开发集群，这样您就可以理解集群任务的不同方面。今天我解释了如何用#Terraform + #Ansible 建立一个 Kafka 集群，这样我们就可以完全控制所有的事情。

**注:**

1.  你应该知道阿帕奇卡夫卡的基本知识，如果不是的话，那就去读一读，或者试试我以前的文章。
2.  你应该知道 Terraform 的基础知识，我不能在这里涵盖这些东西。
3.  你应该知道 AWS EC2，VPC 和标签。
4.  翻译知识也是必需的。

**在 AWS 云中应该做一些事情，这样我们就可以从 Terraform 代码开始。**

1.  创建 IAM 用户并保存其凭证，如 access_key 和 secret_key。
2.  创建 EC2 ssh-key 对，稍后将用于 Ansible。
3.  每个 VPC 子网都应该有标签“type=public/private”。
4.  应在所有 AZ 中创建子网，并且其中不应有空白。
5.  这些文章中分享的所有示例都会随着时间的推移而改变，所以请查看 [GitHub 以获取最新代码](https://github.com/116davinder/kafka-cluster-ansible/tree/master/terraform/aws)。

让我们从 Terraform 代码开始，试着理解它。

应该有一个[**Provider . TF**](https://github.com/116davinder/kafka-cluster-ansible/blob/master/terraform/aws/provider.tf)**指定使用哪个 provider 插件和地区。**

```
terraform {
  required_version        = ">= 0.12"
}provider aws {   
    region                = var.region
}
```

**[**data . TF**](https://github.com/116davinder/kafka-cluster-ansible/blob/master/terraform/aws/data.tf)**/收集事实****

```
data "aws_ami" "amazon_ami" {
most_recent               = true
owners                    = ["amazon"]filter {
   name                   = "owner-alias"
   values                 = ["amazon"]
 }filter {
   name                   = "name"
   values                 = ["amzn2-ami-hvm*"]
 }

}data "aws_availability_zones" "available" {
  state                   = "available"
}# it should be private subnets in production environment.
data "aws_subnet" "public_subnet" {
  count                   = var.kafka_nodes
  vpc_id                  = var.vpc_id
  filter {
    name                  = "tag:type"
    values                = ["public"]
  }
  availability_zone       = data.aws_availability_zones.available.names[ count.index % length(data.aws_availability_zones.available.names) ]}
```

**data.tf 有趣的部分是“aws_subnet”。它将找到所有带有“tag: type=public”的“可用”子网。**

```
data.aws_availability_zones.available.names[ count.index % length(data.aws_availability_zones.available.names) ]
```

**上面这段代码有很多东西，它会在每个 AZ 中找到子网。这将用于在所有 Az 上平均分配 Kafka 节点。**

****例子:****

****地区:**EU-中央-1
AZ ' s 带子网:3
卡夫卡节点数:9
分布:每个 AZ 上 3 个卡夫卡节点。**

****地区:**us-east-1
**AZ ' s with Subnet:**5
**卡夫卡节点数:** 10
**分布**:每个 AZ 上 2 个卡夫卡节点。**

****所有#Terraform 变量**:[Kafka-cluster-ansi ble/terra form/AWS/var . TF](https://github.com/116davinder/kafka-cluster-ansible/blob/master/terraform/aws/var.tf)**

**在创建 EC2 机器之前，我们需要创建一些东西。**

1.  **由于 Kafka 需要大量的持久存储，所以我将使用 [**AWS EBS**](https://github.com/116davinder/kafka-cluster-ansible/blob/master/terraform/aws/ebs.tf) 。**

```
resource "aws_ebs_volume" "kafka" {
  count                   = var.kafka_nodes
  availability_zone       = data.aws_availability_zones.available.names[ count.index % length(data.aws_availability_zones.available.names) ]
  size                    = var.kafka_volume_sizetags = {
    Name                  = "kafka-data-vol-${var.env}-${count.index}"
    Env                   = var.env
    Owner                 = "Terraform"
  }}
```

**2.我需要创建一个安全组。**

****请务必更改以下代码，因为它允许从任何地方访问 SSH / Kafka 端口**。**

```
resource "aws_security_group" "kafka_sg" {
  name                    = "kafka-sg-${var.env}"
  description             = "Allow kafka traffic"
  vpc_id                  = var.vpc_idingress {
    from_port             = 22
    to_port               = 22
    protocol              = "tcp"
    cidr_blocks           = ["0.0.0.0/0"]
  }ingress {
    from_port             = 0
    to_port               = 0
    protocol              = "-1"
    self                  = true
  }ingress {
    from_port             = 9092
    to_port               = 9092
    protocol              = "tcp"
    cidr_blocks           = ["0.0.0.0/0"]
  }egress {
    from_port             = 0
    to_port               = 0
    protocol              = "-1"
    cidr_blocks           = ["0.0.0.0/0"]
  }tags                    = {
    Name                  = "kafka-sg-${var.env}"
    Env                   = var.env
    Owner                 = "Terraform"
  }
}
```

**3.我们还需要创建 [**IAM EC2 概要文件**](https://github.com/116davinder/kafka-cluster-ansible/blob/master/terraform/aws/iam.tf) ，以便 EC2 机器可以在需要时将其日志写入 **AWS CloudWatch** 。**

```
data "aws_iam_policy" "cloudwatchAgent" {
  arn = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
}resource "aws_iam_role" "Kafka-CloudWatchAgentServerRole" {
  name = var.ec2_cloudwatch_roleassume_role_policy = <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
    {
        "Action": "sts:AssumeRole",
        "Principal": {
        "Service": "ec2.amazonaws.com"
        },
        "Effect": "Allow",
        "Sid": ""
    }
    ]
}
EOFtags = {
    Owner                 = "Terraform"
  }
}resource "aws_iam_role_policy_attachment" "Kafka-CloudWatchAgentServerRole" {
  role       = var.ec2_cloudwatch_role
  policy_arn = data.aws_iam_policy.cloudwatchAgent.arndepends_on = [aws_iam_role.Kafka-CloudWatchAgentServerRole, data.aws_iam_policy.cloudwatchAgent]
}resource "aws_iam_instance_profile" "Kafka-CloudWatchAgentServerRole-Profile" {
  name = var.ec2_cloudwatch_role
  role = aws_iam_role.Kafka-CloudWatchAgentServerRole.name
}
```

**现在是时候为卡夫卡创造 [EC2 机器了。](https://github.com/116davinder/kafka-cluster-ansible/blob/master/terraform/aws/kafka.tf)**

1.  **我也使用 EBS 卷作为根卷。**
2.  **现在不需要详细的监控。**

```
resource "aws_instance" "kafka" {
  count                   = var.kafka_nodesami                     = data.aws_ami.amazon_ami.id
  instance_type           = var.instance_type
  key_name                = var.key_name
  subnet_id               = element(data.aws_subnet.public_subnet.*.id, count.index)
  vpc_security_group_ids  = ["${aws_security_group.kafka_sg.id}"]

  root_block_device {
    volume_type           = "gp2"
    volume_size           = var.kafka_root_volume_size
  }tags = {
    Name                  = "kafka-${var.env}-${count.index}"
    Env                   = var.env
    Owner                 = "Terraform"
    Software              = "Apache Kafka"
  }volume_tags = {
    Owner                 = "Terraform"
  }monitoring              = falseavailability_zone       = data.aws_availability_zones.available.names[ count.index % length(data.aws_availability_zones.available.names) ]
  iam_instance_profile    = var.ec2_cloudwatch_role
  depends_on              = [aws_security_group.kafka_sg,aws_iam_instance_profile.Kafka-CloudWatchAgentServerRole-Profile]}
```

**一旦 EC2 代码准备就绪，[我们还需要附加 EBS](https://github.com/116davinder/kafka-cluster-ansible/blob/master/terraform/aws/kafka.tf#L34-L40) 。**

```
resource "aws_volume_attachment" "kafka_ebs_attach" {
  count                   = var.kafka_nodes
  device_name             = var.kafka_ebs_attach_location
  volume_id               = element(aws_ebs_volume.kafka.*.id, count.index)
  instance_id             = element(aws_instance.kafka.*.id, count.index)
  force_detach            = true
}
```

**我们还需要 EC2 机器的出口 IP，所以[输出。
**注意:如果使用私有子网，也应该更改。**](https://github.com/116davinder/kafka-cluster-ansible/blob/master/terraform/aws/ouput.tf)**

```
output public_ip {
  value                   = aws_instance.kafka[*].public_ip
  sensitive               = false
}
```

****如何运行 Terraform 代码？****

****克隆项目:**[116 dav inder/Kafka-cluster-ansi ble](https://github.com/116davinder/kafka-cluster-ansible)或[下载发布](https://github.com/116davinder/kafka-cluster-ansible/releases)。
将文件夹切换到**Kafka-cluster-ansi ble/terra form/AWS。
运行地形命令** : init / apply。
或
**跟随本自述:**[116 dav inder/Kafka-cluster-ansi ble/terraform/Readme . MD](https://github.com/116davinder/kafka-cluster-ansible/blob/master/terraform/readme.md)**

**Terraform 将为我们创造所有的裸资源。现在是 Ansible 再次发光的时候了。**

**由于 EC2 机器上的 AWS 云，我们必须做一些预设置。**

1.  **安装 Python 3 及其依赖项。**
2.  **安装 AWS 云观察日志代理。**
3.  **将 EBS 卷配置到分区。**

**[**库存**](https://github.com/116davinder/kafka-cluster-ansible/blob/master/inventory/development/cluster-aws.ini)**

****参考消息:**我没有使用动态库存。**

```
[all:vars]
ansible_user=ec2-user
ansible_connection=ssh 
ansible_become_method=sudo 
ansible_become=true
ansible_ssh_private_key_file=~/.ssh/davinder-test-terraform.pem[clusterNodes]
18.159.111.208[clusterAddNodes][clusterRemoveNodes][kafka-mirror-maker][kafka-mirror-maker-remove-nodes]
```

****所需的额外变量是****

```
# Variables Only for AWS Based Cluster
aws_kafka_ec2_region: "eu-central-1"
aws_kafka_ebs_device: /dev/xvdc
aws_kafka_ebs_device_fs: xfs
aws_kafka_ebs_device_mount_location: "{{ kafkaInstallDir }}"
```

**我已经创建了一个非常简单可行的剧本: [clusterAwsPreSetup.yml](https://github.com/116davinder/kafka-cluster-ansible/blob/master/clusterAwsPreSetup.yml)**

```
---- name: install xfs untils
  package:
    name: xfsprogs
    state: present
  ignore_errors: true- name: check filesystem on given device
  command: file -s "{{ aws_kafka_ebs_device }}"
  register: kafka_ebs_device_status- name: create filesystem on given device
  filesystem:
    fstype: "{{ aws_kafka_ebs_device_fs }}"
    dev: "{{ aws_kafka_ebs_device }}"
  when: kafka_ebs_device_status.stdout | regex_search(' data')- name: create kafka ebs mount dir
  file:
    path: "{{ aws_kafka_ebs_device_mount_location }}"
    state: directory- name: mount kafka ebs volume
  mount:
    path: "{{ aws_kafka_ebs_device_mount_location }}"
    src: "{{ aws_kafka_ebs_device }}"
    fstype: xfs
    state: mounted
```

**行动手册的其余部分现在将按预期工作。**请阅读**[**readme . MD**](https://github.com/116davinder/kafka-cluster-ansible/blob/master/Readme.md)**。****

**旅程将在下一个主题继续(Oracle 云上的 Terraform)**