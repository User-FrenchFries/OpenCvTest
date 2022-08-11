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

