### 大致简介

此Android工程嵌入OpenCv库，调试使用其中的多个算法进行图像增强的测试，具体结果如下

由于测试图像较多，单以一张效果进行展示（图片由网上示例摘抄，如有侵权，请告知，我方会自行删除，并对其带来的影响深表歉意。）

### 受测试的算法如下：

#### 1. 直方图均衡化

此方法用于增强图像对比度，可以使过亮或过暗的图片变得更加柔和。

#### 2. 对比度受限直方图均衡化（CLAHE）

此方法为直方图均衡化的优化算法。

#### 3. 拉普拉斯算子增强

此方法用于整体偏暗图像的增强，变亮。

#### 4. 对数变换算法

此方法可以将图像的低灰度值部分扩展，显示出低灰度部分更多的细节，将其高灰度值部分压缩，减少高灰度值部分的细节，从而达到强调图像低灰度部分的目的。

#### 5. 伽马变换

此方法原理同对数变化，γ大于1可强调高灰度值，γ小于1可强调低灰度值。

### 具体效果如下（展示手机模拟器运算结果）

#### 1. 原始图像

![](https://github.com/User-FrenchFries/OpenCvTest/blob/master/picResult/origin.png)

#### 2. 直方图均衡化

![](https://github.com/User-FrenchFries/OpenCvTest/blob/master/picResult/he.png)

#### 3. CLAHE

默认情况

![](https://github.com/User-FrenchFries/OpenCvTest/blob/master/picResult/clahe_default.png)

自主调试裁剪大小（7.7）

![](https://github.com/User-FrenchFries/OpenCvTest/blob/master/picResult/clahe_7.7.png)

(0.0)

![](https://github.com/User-FrenchFries/OpenCvTest/blob/master/picResult/clahe_0.0.png)

#### 4. 拉普拉斯

![](https://github.com/User-FrenchFries/OpenCvTest/blob/master/picResult/lapras.png)

#### 5. 对数变化

![](https://github.com/User-FrenchFries/OpenCvTest/blob/master/picResult/log.png)

#### 6. 伽马变化

![](https://github.com/User-FrenchFries/OpenCvTest/blob/master/picResult/gamma.png)

### 各种优化方法个人理解

#### 1. 直方图均衡化---均衡化图像中的像素，提高图像对比度

##### 第一个概念

**图片对比度指的是一幅图片中最亮的白和最暗的黑之间的反差大小。**常用的定量度量方法是**Michelson对比度**：
$$
C=Imax−Imin/Imax+IminC
$$

- 当一幅图像最白和最黑像素灰度都是128时，图像对比度最低，C=0；
- 当一幅图像最白像素灰度=255，最黑像素灰度=0时，图像对比度最高，C=1.0。

##### **第二个概念**

**根据信息论, 信息的熵越大, 包含的信息也就越多, 熵的计算公式如下:**
$$
H=-\sum_{i=0}^{n}p(x_i)log_2(p(x_i)) \tag{1}
$$
只有当 p ( x i ))均匀分布时, 熵的值最大. 对应到图像上, 当图像直方图均匀分布时, 图像对比度最大. 

因此可以使用此方法将图像中的像素进行处理（映射处理），提高图像的对比度（采用累积分布函数）

##### 采用累积分布函数原因

此方法是为了将像素均衡化，映射完毕后要保证原来大小关系不变（原来的亮块映射后仍然是亮块）；映射后的像素大小还是要在原有范围内。因此就使得该映射函数值域为[0,1]，并且是单调增函数

##### **具体做法如下**

1. 统计计算原有图像中像素级及其对应像素块个数
2. 计算其出现概率，进而计算得到累积概率，再经过映射函数得到最终的调整像素
3. 组装最终的像素块为新的图片

#### 2. CLAHE

##### 优化原因

此种方法是为了弥补上述处理方式的缺陷。

1. 直方图均衡是全局的, 对图像局部区域存在过亮或者过暗时, 效果不是很好;
2. 直方图均衡会增强背景噪声（黑色区块出现意外的雪花斑点）

##### 解决思想

对应的解决思想如下：

1. 全局性问题：将图像划分为小区块，分区块进行处理，减小全局性影响（AHE方法已经优化）
2. 噪声问题：主要背景增强太过了, 因此对对比度进行限制

##### 具体步骤

1. 图像分块。
2. 分块进行像素分布修正（这时候就由CLAHE方法中的clipLimit参数限制整个区块的映射后像素分布情况，截断超出比例部分并进行均匀化重排）
3. 将分块图形整合为整副图像。

#### 3. 拉普拉斯

##### 概念科普

由于拉普拉斯是一种微分算子，它的应用可增强图像中灰度突变的区域，减弱灰度的缓慢变化区域。因此，锐化处理可选择拉普拉斯算子对原图像进行处理，产生描述灰度突变的图像，再将拉普拉斯图像与原始图像叠加而产生锐化图像。

下面这个内核：

|   0    |   -1   |   0    |
| :----: | :----: | :----: |
| **-1** | **5**  | **-1** |
| **0**  | **-1** | **0**  |

它等于 1 减去拉普拉斯内核（也就是原始图像减去它的拉普拉斯图像）。

#### 4. 对数变化

##### 概念科普

由于对数曲线在像素值较低的区域斜率大，在像素值较高的区域斜率较小，所以图像经过对数变换后，较暗区域的对比度将有所提升，所以就可以增强图像的暗部细节。

对数变换可以将图像的低灰度值部分扩展，显示出低灰度部分更多的细节，将其高灰度值部分压缩，减少高灰度值部分的细节，从而达到强调图像低灰度部分的目的。变换方法：
$$
S = Clog(1+r)-----C为常数，r>=0
$$

#### 5. 伽马变化

##### 概念科普

最简单的伽马校正运算是由一个幂函数的公式所定义的，公式如下
$$
S = Cr^\gamma    -----  S为输出，r为输出，\gamma为变换值
$$
其中，c为一个常数系数，控制整条伽马曲线的倾斜程度，一般默认为1。幂函数的幂规定大于0，有一个关键的分界点1。在【0，1】区间伽马曲线呈上凸形态，在【1，无穷】区间呈下凸形态。从伽马曲线图可以看出：

- 在【0，1】区间，γ 值小于1时，会拉伸图像中灰度级较低的区域，同时会压缩灰度级较高的部分，与此对应的变化是图像的暗部细节会得到提升。
- 在【1，无穷】，γ 值大于1时，会拉伸图像中灰度级较高的区域，同时会压缩灰度级较低的部分，这样处理和的结果是图像的对比度得到明显提升。