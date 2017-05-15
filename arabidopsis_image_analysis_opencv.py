# -*- coding: utf-8 -*-
"""
Created on Tue May 09 12:03:21 2017

@author: THERIN J. YOUNG
"""

from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
import argparse #used for image rotation
import imutils

#set working directory where images are located
os.chdir('c:/Users/theri/Dropbox/Other Projects/Images/Camera876')


#Load image
img = cv2.imread('CAM876_2017-03-31-14-01.JPG')



#Define variables
x = 520
y = 510


#loop over the rotation angles (counterclockwise)
for angle in np.arange (0, 360, 0.4):
    rotated = imutils.rotate(img, angle)


    #create window to display rotating image    
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 1000, 1000)
    cv2.imshow('image', rotated)

    #press the "a" key once image is rotated to desired position    
    if cv2.waitKey(1000) == ord('a'):

        #crop the rotated image and save it using the imwrite command
        crop_img = rotated[409:3585, 1473:4697] #coordinate system is y1:y2, x1:x2
        cv2.imwrite('image3.jpg', crop_img)
        
        cv2.namedWindow('cropped', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('cropped', 800, 800)
        cv2.imshow("cropped", crop_img)
        cv2.waitKey(3000)
        
        #Crop out tray 1 and write it to image file
        tray1 = crop_img[9:3161, 13:1581]
        cv2.imwrite('tray1.jpg', tray1)
        
        #crop out tray 2 and write it to image file
        tray2 = crop_img[9:3176, 1641:3217]
        cv2.imwrite('tray2.jpg', tray2)
      
        cv2.destroyAllWindows()
        
        
        ############################################################################################################
        #SEPARATE POTS IN TRAY 1 and change color spaces to hsv and lab
        ############################################################################################################
        
        #define range of green color in HSV
        
        lower_green = np.array([25,100,100])
        upper_green = np.array([60,255,255])
        
        
        #pots 1-3
        
            
        p11 = tray1[1033+(3*y):1553+(3*y), 521-x:1041-x]
        p11_hsv = cv2.cvtColor(p11, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(p11_hsv, lower_green, upper_green)
        result = cv2.bitwise_and(p11_hsv,p11_hsv,mask = mask)
        
        #noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel, iterations = 4)
        
        #sure background area
        closing1 = cv2.dilate(opening,kernel,iterations=7)
        
        cv2.imshow('result', closing1)
        
        
        cv2.waitKey(2000)
        
        
        #################################################################################################
        ##############################THE REMAINING POTS OF TRAY 1#######################################
        #################################################################################################
        p12 = tray1[1033+(3*y):1553+(3*y), 521:1041]
        p13 = tray1[1033+(3*y):1553+(3*y), 521+x:1041+x]
        
        #pots 4-6
        
        p21 = tray1[1033+(2*y):1553+(2*y), 521-x:1041-x]
        p22 = tray1[1033+(2*y):1553+(2*y), 521:1041]
        p23 = tray1[1033+(2*y):1553+(2*y), 521+x:1041+x]
        
        #pots 7-9
        
        p31 = tray1[1033+y:1553+y, 521-x:1041-x]
        p32 = tray1[1033+y:1553+y, 521:1041]
        p33 = tray1[1033+y:1553+y, 521+x:1041+x]
        
        #pots 10-12
        
        p41 = tray1[1033:1553, 521-x:1041-x]
        p42 = tray1[1033:1553, 521:1041]
        p43 = tray1[1033:1553, 521+x:1041+x]
        
        #pots 13-15
        
        p51 = tray1[1033-y:1553-y, 521-x:1041-x]
        p52 = tray1[1033-y:1553-y, 521:1041]
        p53 = tray1[1033-y:1553-y, 521+x:1041+x]
        
        #pots 16-18
        
        p61 = tray1[1033-(2*y):1553-(2*y), 521-x:1041-x]
        p62 = tray1[1033-(2*y):1553-(2*y), 521:1041]
        p63 = tray1[1033-(2*y):1550-(2*y), 521+x:1041+x]
        
        # show 18 pots from tray 1 in one window
        
        plt.subplot(6,3,1),plt.imshow(p61,'Greens'),plt.title('pot16')
        plt.axis("off")
        plt.subplot(6,3,2),plt.imshow(p62,'Greens'),plt.title('pot17')
        plt.axis("off")
        plt.subplot(6,3,3),plt.imshow(p63,'Greens'),plt.title('pot18')
        plt.axis("off")
        plt.subplot(6,3,4),plt.imshow(p51,'Greens'),plt.title('pot13')
        plt.axis("off")
        plt.subplot(6,3,5),plt.imshow(p52,'Greens'),plt.title('pot14')
        plt.axis("off")
        plt.subplot(6,3,6),plt.imshow(p53,'Greens'),plt.title('pot15')
        plt.axis("off")
        plt.subplot(6,3,7),plt.imshow(p41,'Greens'),plt.title('pot10')
        plt.axis("off")
        plt.subplot(6,3,8),plt.imshow(p42,'Greens'),plt.title('pot11')
        plt.axis("off")
        plt.subplot(6,3,9),plt.imshow(p43,'Greens'),plt.title('pot12')
        plt.axis("off")
        plt.subplot(6,3,10),plt.imshow(p31,'Greens'),plt.title('pot7')
        plt.axis("off")
        plt.subplot(6,3,11),plt.imshow(p32,'Greens'),plt.title('pot8')
        plt.axis("off")
        plt.subplot(6,3,12),plt.imshow(p33,'Greens'),plt.title('pot9')
        plt.axis("off")
        plt.subplot(6,3,13),plt.imshow(p21,'Greens'),plt.title('pot4')
        plt.axis("off")
        plt.subplot(6,3,14),plt.imshow(p22,'Greens'),plt.title('pot5')
        plt.axis("off")
        plt.subplot(6,3,15),plt.imshow(p23,'Greens'),plt.title('pot6')
        plt.axis("off")
        plt.subplot(6,3,16),plt.imshow(closing1,'Greens'),plt.title('pot1')
        plt.axis("off")
        plt.subplot(6,3,17),plt.imshow(p12,'Greens'),plt.title('pot2')
        plt.axis("off")
        plt.subplot(6,3,18),plt.imshow(p13,'Greens'),plt.title('pot3')
        plt.axis("off")
       
        plt.show()
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        

    
        
        
        
        
        
        
        
        
        
        
        
        break
    



    



    
 
        
        
        
    

    
