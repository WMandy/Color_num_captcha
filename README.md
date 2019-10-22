# -
验证码样式如下
# 验证码
![](https://github.com/WMandy/Color_num_captcha/blob/master/000000.png)

步骤： 
1.根据下方文字提示在图中找到对应颜色的数字  
2.分割数字（膨胀腐蚀，寻找最小边界框）  
3.分割出的单数字图像归一化，上knn算法，bn_label文件夹是训练集  
