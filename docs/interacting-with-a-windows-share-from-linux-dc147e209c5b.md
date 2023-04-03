# 从 Linux 与 Windows 共享交互

> 原文：<https://medium.com/analytics-vidhya/interacting-with-a-windows-share-from-linux-dc147e209c5b?source=collection_archive---------0----------------------->

问题:我需要在 Linux 文件系统和 Windows 文件共享之间传输文件。

解决方案:使用`mount`和`smbclient`让事情运转起来。

```
smbclient //host/share$ -U domain/user%password --max-protocol=SMB2_24 -c 'get backup\backup.txt backup/backup.txt'smbclient //host/share$ -U domain/user%password --max-protocol=SMB2_24 -c 'put backup/backup.txt backup\backup.txt'mount -t cifs //host/share$ /mnt/share -o username=user,domain=domain,vers=2.1
```