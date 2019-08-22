# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 10:52:31 2019

@author: mandi_wang
"""
import os 
import cv2
import numpy as np

def distance(img1,img2):
    h,w=img1.shape
    d=0
    for i in range(h):
        for j in range(w):
            if img1[i,j]!=img2[i,j]:
                d+=1
    return d

def reco_color(file,img,lower,upper):
#    frame=img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(gray, lower, upper)
#    cv2.imshow('Mask', mask)
#    cv2.waitKey(0)
#    res = cv2.bitwise_and(frame, frame, mask=mask)
#    cv2.imshow('Result', res)
#    cv2.waitKey(0)
    ret, binary = cv2.threshold(mask,200,255,cv2.THRESH_BINARY) 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(4, 5))
    dilated = cv2.dilate(binary,kernel,iterations = 1)
#    cv2.imshow('Mask', dilated)
#    cv2.waitKey(0)
    contours, hierarchy = cv2.findContours(dilated,3,3)
    l=[]
    #print(contours)
    for cnt in contours:
    
    # 最小的外接矩形
        x, y, w, h = cv2.boundingRect(cnt)
        tmp=[]
        for i in x,y,w,h:
            tmp.append(i)
           
        l.append(tmp)
    l.sort()
    flag = 1
    test_imgs=[]
    for i in l:
        x,y,w,h=i[0],i[1],i[2],i[3]
        
        if  w*h >= 100:
            #print('Selected',(x,y,w,h))
            # 显示图片
            img_split=img[y-1:y+h+1, x-1:x+w+1]
            cv2.imwrite('./split/'+file[0:6]+'_'+'%s.jpg'%flag,img_split)
            test_imgs.append(img_split)
#            gray_split = cv2.cvtColor(img_split, cv2.COLOR_BGR2GRAY)
#            ret, binary1 = cv2.threshold(gray_split,200,255,cv2.THRESH_BINARY) 
#            cv2.imwrite('./bn_split/'+file[0:6]+'_'+'%s.jpg'%flag,binary1)
            flag+=1
    return test_imgs

p='./bn_label'
files=sorted(os.listdir(p))
k=5

##制作数据集
dataset={}  
for i in range(0,10):
    ch_images=[]
    for j in range(0,9):
        img=cv2.imread(os.path.join(p,'%s.jpg'%(str(i*10+j)).zfill(2)),0)
        img = cv2.resize(img,(20,20),interpolation = cv2.INTER_AREA) 
        ret, img = cv2.threshold(img,200,255,cv2.THRESH_BINARY) 
        ch_images.append(img)
    dataset[i]=ch_images
    
p=r'C:\Users\mandi_wang\Desktop\num_color\data_images\image/'
files=os.listdir(p)
for file in files:
    img=cv2.imread(os.path.join(p,file))

    if img[39,52][0]==0:
        a='Yellow'
    elif img[38,53][0]==14:
        a='Red'
        
    if a=='Yellow':
        lower=np.array([0,160,160])  #黄色
        upper=np.array([40,255,255])
        test_imgs=reco_color(file,img,lower,upper)
        
    if a=='Red':
        lower=np.array([0,160,160])  #红色
        upper=np.array([200,255,255])
        test_imgs=reco_color(file,img,lower,upper)


#knn
    nums=[]
    for img1 in test_imgs:
    #    img1=cv2.imread(os.path.join(test_p,file))
        img1 = cv2.resize(img1,(20,20),interpolation = cv2.INTER_AREA) 
        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray,200,255,cv2.THRESH_BINARY) 
        d=[]
        for key,value in dataset.items():
            
            for v in value: 
                d_t=distance(binary,v)
                d.append((d_t,key))
                
        sort_d=sorted(d)   
        vote_count={}
        for i in range(k):
            vote_label=sort_d[i][1]
            vote_count[vote_label]=vote_count.get(vote_label,0)+1
       # print(vote_count)
        
        max_count=0
        for key,value in vote_count.items():
            if vote_count[key]>max_count:
                max_count=value
                max_label=key
        nums.append(str(max_label))              
        reco=''.join(nums)
    print('识别结果：',reco,'\n','-'*50,'\n') 




