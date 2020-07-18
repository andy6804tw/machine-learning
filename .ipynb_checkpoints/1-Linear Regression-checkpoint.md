## 線性迴歸(Linear Regression)
線性回歸（Linear regression）是統計上在找多個自變數(independent variable)和依變數(dependent variable)之間的關係建出來的模型。只有一個自變數和一個依變數的情形稱為簡單線性回歸(Simple linear regression)，大於一個自變數的情形稱為多元回歸(multiple regression)。

簡單線性回歸: y=β0+β1x
β0：截距(Intercept)，β1：斜率(Slope)為 x變動一個單位y變動的量，如下圖:

![](https://imgur.com/OvYzfGw.jpg)

## 範例實作
線性回歸簡單來說，就是將複雜的資料數據，擬和至一條直線上，就能方便預測未來的資料。

先從簡單的線性回歸舉例，![](https://chart.googleapis.com/chart?cht=tx&chl=y%20%3D%20ax%20%2B%20b) ，![](https://chart.googleapis.com/chart?cht=tx&chl=a) 稱為斜率，![](https://chart.googleapis.com/chart?cht=tx&chl=b) 稱為截距。以下範例我們假設 `a=3  b=15`，


```py
# imports
import numpy as np
import matplotlib.pyplot as plt

# generate random data-set
np.random.seed(0)
noise = np.random.rand(100, 1)
x = np.random.rand(100, 1)
y = 3 * x + 15 + noise
# y=ax+b Target function  a=3, b=15


# plot
plt.scatter(x,y,s=10)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```
