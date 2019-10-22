# -
破解比较简单的纯数字验证码，没必要爬上千张样本再搞个cnn网络训练，用常规的传统算法效果通常都ok。
## 验证码样式
![](https://github.com/WMandy/Color_num_captcha/blob/master/example_images/000000.png)

步骤： 
1.根据下方文字提示在图中找到对应颜色的数字  
2.分割数字（膨胀腐蚀，寻找最小边界框）  
3.分割出的单数字图像归一化，上knn算法，算法中的距离定义为，测试样本与训练样本灰度值不同的像素点的数量  
（bn_label文件夹是训练集）  

 ## 部分训练集图片样式
 ![](https://github.com/WMandy/Color_num_captcha/blob/master/example_images/20191022144611.png)
