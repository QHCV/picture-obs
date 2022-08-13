#  Pytorch开发环境安装（Anaconda+Cuda+pycharm）

[TOC]

## 0.pytorch简介

PyTorch是一个基于Torch的Python开源机器学习库，用于自然语言处理等应用程序，目前是主流的深度学习框架，它主要由Facebook的人工智能研究小组开发。

> pytorch是Python的一个深度学习包，有两个功能：
>
> -  具有强大的GPU加速的张量计算（如NumPy） 
> - 包含自动求导系统的的深度神经网络

## 1.Anaconda安装及基本配置

### 1.1 Anaconda安装

> 官方下载网址:[Anaconda | Anaconda Distribution](https://www.anaconda.com/products/distribution#)
>
> 国内下载地址(速度较快)：[Index of /anaconda/archive/ | 清华大学开源软件镜像站 | Tsinghua Open Source Mirror](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/) ,选择最新版本下载。

![image-20220808144326538](https://raw.githubusercontent.com/QHCV/picture-obs/main/2022.08/202208081443609.png)

双击下载程序点击安装。

![image-20220810204624237](https://raw.githubusercontent.com/QHCV/picture-obs/main/2022.08/202208102046278.png)

![image-20220810204714243](https://raw.githubusercontent.com/QHCV/picture-obs/main/2022.08/202208102047279.png)

![image-20220810205831092](https://raw.githubusercontent.com/QHCV/picture-obs/main/2022.08/202208102058133.png)

安装完毕后，conda还是不可用，这是因为需要添加环境变量。

![image-20220808172313522](https://raw.githubusercontent.com/QHCV/picture-obs/main/2022.08/202208081723575.png)

### 1.2 配置环境变量

如果在安装过程中选择的是just me,就可以选择自动添加环境变量。

> 配置以下路径
>
> - XX\anaconda
> - XX\anaconda\Scripts\
> - XX\anaconda\Library\bin
> - XX\anaconda\Lib\site-packages
> - XX\anaconda\Library\mingw-w64\bin

![image-20220808172849917](https://raw.githubusercontent.com/QHCV/picture-obs/main/2022.08/202208081728089.png)

![image-20220808172816761](https://raw.githubusercontent.com/QHCV/picture-obs/main/2022.08/202208081728823.png)

**添加完成后确定。**

![image-20220809101845363](https://raw.githubusercontent.com/QHCV/picture-obs/main/2022.08/202208091018411.png)

```
查看版本，输入
conda --version
查看基本信息，输入
conda info
```

![image-20220809102030231](https://raw.githubusercontent.com/QHCV/picture-obs/main/2022.08/202208091020277.png)

**conda安装成功！**

### 1.3 添加清华源国内镜像

：由于国内使用conda网速较慢,需要配置清华源镜像。

[anaconda | 镜像站使用帮助 | 清华大学开源软件镜像站 | Tsinghua Open Source Mirror](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)

打开上面的网址，有具体说明。

C盘下找到下面的文件，以笔记本打开。

![image-20220813235414023](https://raw.githubusercontent.com/QHCV/picture-obs/main/2022.08/image-20220813235414023.png)

复制以下内容到.condarc中

```
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

![image-20220813235550264](https://raw.githubusercontent.com/QHCV/picture-obs/main/2022.08/image-20220813235550264.png)

最后保存就可以了

## 2.Cuda安装

查看电脑是否安装了cuda

```
#cmd中输入如下命令，查看电脑是否安装了Cuda
nvidia-smi
```

![image-20220810205039421](https://raw.githubusercontent.com/QHCV/picture-obs/main/2022.08/202208102050477.png)

如果显示‘nvidia-smi‘ 不是内部或外部命令，参考这篇解决：https://blog.csdn.net/QH2107/article/details/126155833

一般情况下都是安装了的，如果没有安装或者需要对应的cuda版本可参考这位博主的教程：

[【CUDA】cuda安装 （windows版）_何为xl的博客-CSDN博客_windows安装cuda](https://blog.csdn.net/weixin_43848614/article/details/117221384)

## 3.环境配置

### 3.1 打开pycharm ->打开终端

- 利用anaconda创建运行环境


```
#命令行输入
conda create -n pytorchS python=3.6
#若需删除环境
conda remove -n your_env_name --all
```

- 激活环境

  ```
  #激活环境
  conda activate pytorchS
  #关闭环境命令
  deactivate pytorchS
  ```
  
  ![image-20220805211211044](https://raw.githubusercontent.com/QHCV/picture-obs/main/2022.08/image-20220805211211044.png)

如果前面是PS，在Pycharm设置中修改Shell为CMD。

![image-20220805211527129](https://raw.githubusercontent.com/QHCV/picture-obs/main/2022.08/image-20220805211527129.png)

### 3.2 配置pycharm运行环境

将Pycharm运行环境配置为刚才创建的环境

![image-20220805211748654](https://raw.githubusercontent.com/QHCV/picture-obs/main/2022.08/image-20220805211748654.png)

在anaconda目录下依次选择到目标文件夹，选择python.exe，确定。

![image-20220805211825616](https://raw.githubusercontent.com/QHCV/picture-obs/main/2022.08/image-20220805211825616.png)

当前环境就是刚才的创建的环境了

![image-20220805212016754](https://raw.githubusercontent.com/QHCV/picture-obs/main/2022.08/image-20220805212016754.png)

### 3.3 安装pytorch

点击[进入PyTorch官网](https://pytorch.org/)，选择合适的安装要求。

- 查看电脑Cuda版本

  > ＣＭＤ中输入：
  >
  > ｎｖｉｄｉａ－ｓｍｉ

  ![image-20220805212554695](https://raw.githubusercontent.com/QHCV/picture-obs/main/2022.08/image-20220805212554695.png)

- 选择对应配置，复制Command

![image-20220805212620035](https://raw.githubusercontent.com/QHCV/picture-obs/main/2022.08/image-20220805212620035.png)

运行命令

![image-20220805213729206](https://raw.githubusercontent.com/QHCV/picture-obs/main/2022.08/image-20220805213729206.png)

使用conda list 或者解释器设置中查看已安装的python包

![image-20220805213917136](https://raw.githubusercontent.com/QHCV/picture-obs/main/2022.08/image-20220805213917136.png)

这时发现在安装包中没有torch包，明明提示pytorch安装成功了。

找了一圈发现是python版本太低了，换了一个python3.9版本的试了一下，可以了。

```python
#测试是否可用
import torch # 如果pytorch安装成功即可导入
print(torch.cuda.is_available()) # 查看CUDA是否可用
print(torch.cuda.device_count()) # 查看可用的CUDA数量
print(torch.version.cuda) #查看Cuda版本
```

> 运行结果：
>
> True
> 1
> 11.6

Pytorch环境安装完毕！



**当Conda安装不成功时，可以离线安装：**

先到下面的网址下载对应版本的pytorch

> https://download.pytorch.org/whl/torch_stable.html
>
> （其中，cu100表示cuda版本是10.0，torch-1.0.0表示torch的版本，cp35表示python的版本是3.5，最后部分用于区分linux和windows系统，注意pytorch版本和torchvision版本需要对应）

下载完成之后使用cmd进入文件存放路径，输入pip install 文件名 即可成功安装。
