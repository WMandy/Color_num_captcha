# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 10:52:31 2019
@author: mandi_wang
"""
import os 
import cv2
import numpy as np
import base64 
from io import BytesIO

def reco_color(img,lower,upper):
#    frame=img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(gray, lower, upper)
    ret, binary = cv2.threshold(mask,200,255,cv2.THRESH_BINARY) 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(4, 5))
    dilated = cv2.dilate(binary,kernel,iterations = 1)

    contours, hierarchy = cv2.findContours(dilated,3,3)
    l=[]
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
            #cv2.imwrite('./split/'+file[0:6]+'_'+'%s.jpg'%flag,img_split)
            test_imgs.append(img_split)
            flag+=1
    return test_imgs

def distance(img1,img2):
    h,w=img1.shape
    d=0
    for i in range(h):
        for j in range(w):
            if img1[i,j]!=img2[i,j]:
                d+=1
    return d

##制作数据集打标签
def mk_label():
    p='./bn_label'
    dataset={}  
    for i in range(0,10):
        ch_images=[]
        for j in range(0,9):
            img=cv2.imread(os.path.join(p,'%s.jpg'%(str(i*10+j)).zfill(2)),0)
            img = cv2.resize(img,(20,20),interpolation = cv2.INTER_AREA) 
            ret, img = cv2.threshold(img,200,255,cv2.THRESH_BINARY) 
            ch_images.append(img)
        dataset[i]=ch_images
    return dataset

def reco_num(img_base64code):  
    image_data = base64.b64decode(img_base64code)
    nparr=np.fromstring(image_data,np.uint8)
    img=cv2.imdecode(nparr,cv2.IMREAD_COLOR)
    #img=cv2.imread(os.path.join(p,file))

    color_word=img[38:51,52:65]
    a=np.sum(color_word)//255
    a='Yellow' if a<300 else 'Red'
        
    if a=='Yellow':
        lower=np.array([0,160,160])  #黄色
        upper=np.array([40,255,255])
        test_imgs=reco_color(img,lower,upper)
        
    if a=='Red':
        lower=np.array([0,160,160])  #红色
        upper=np.array([200,255,255])
        test_imgs=reco_color(img,lower,upper)
    #knn
    k=5
    nums=[]
    for img1 in test_imgs:
    #    img1=cv2.imread(os.path.join(test_p,file))
        img1 = cv2.resize(img1,(20,20),interpolation = cv2.INTER_AREA) 
        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray,200,255,cv2.THRESH_BINARY) 
        d=[]
        dataset=mk_label()
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
    return reco

img_base64code='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'
a=reco_num(img_base64code)
print(a)

