#-*- coding:utf-8 -*-
from __future__ import division
from __future__ import print_function
import ocr
import cv2
import time
import os
import numpy as np
from PIL import Image
import sys
reload(sys)
sys.setdefaultencoding('utf8')

class ImageProcess:
    
    def text_region_run(self, img):
        
        total_image, total_width, img_framed, detect_time, detect_number = ocr.get_text_region(img)
        
        return total_image, total_width, img_framed, detect_time, detect_number
    
    def densenet_recognition_run(self, img):
        
        results = ocr.densenet_recognition(img)
        
        return results
        
if __name__ == '__main__':
    
    video_text_region = []
    video_text_width = []
    
    img_handle = ImageProcess()
    cap = cv2.VideoCapture('/home/vcaadmin/zhuangwu/huang/scene_text_recognition/video/aaaa_seg_10.ts')
    
    if cap.isOpened():
     
        ret, img = cap.read()
        frame_id = 0
        height, width, depth = img.shape
        
        while ret:
            if frame_id % 500 == 0:
                t0 = time.time()
                img_temp = img.copy()
                img_rgb = cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB)
                img_cut = img_rgb[int(5/6*height):height,0:width]              
                
                total_image, total_width, img_framed, detect_time, detect_number = img_handle.text_region_run(img_cut) 
                
                if len(total_image) > 0:
                    for i in range(len(total_image)):
                        video_text_region.append([total_image[i],frame_id])
                        video_text_width.append([total_width[i],frame_id])
                t1 = time.time()
                print(frame_id,'detect_time:{:.3f}'.format(detect_time), 'detect_number:{}'.format(detect_number),)
#            if frame_id == 800:
#                
#                break
            
            ret, img = cap.read()
            frame_id += 1
        cap.release()
    for index, _ in enumerate(video_text_width):
        video_text_width[index].append(index)
    video_text_width.sort()
    for term in video_text_width:
        #print(term)
        results = img_handle.densenet_recognition_run(video_text_region[term[-1]][0])
        #print(results)
    