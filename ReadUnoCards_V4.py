import argparse
import cv2
import os
import numpy as np
import glob
import array as arr
import pandas as pd
import math
from turtle import left, right
from multiprocessing.connection import wait
from operator import index
from matplotlib import cm
from matplotlib import pyplot as plt
from skimage.filters import threshold_local
from mpl_toolkits.mplot3d import Axes3D

### The user will input the parameters needed to run the script, 
##something like python RealUnoCards_Shape_V3.py --mode livecam --path images/ --camdevice 0; in particular:
##  --camlive is boolean, used to decide whether to recognise cards from livestream or from the testing folder
##  --addcards is boolean, used in case the user wants to train more cards
##  --camdevice is boolean, used in case the user wants to use an external camera.
##  --src is the source directory where the pictures are stored.

parser = argparse.ArgumentParser(description='list of arguments for reading UNO cards', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-t', '--testclass', action='store_true', help='Choose this option to test the same cards used for training. /n'
                    +'If not chosen, the system will run on Run on live camera by default')
parser.add_argument('-g', '--plotgaussian', action='store_true', help='Choose this option to plot the average and std-dev for each color')
parser.add_argument('-c', '--addcards', action='store_true', help='Add cards for testing')
parser.add_argument('-e', '--extcam', action='store_true', help='Choose this option when using a USB external camera')
parser.add_argument('-s', '--setsize', action='store_true', help='Size to pixel setup, using a 2 cm diameter coin as reference')
parser.add_argument('-d', '--debugmode', action='store_true', help='Activate this to show data prints and morphological transformations')
parser.add_argument('imglocation', help='Source location')
args = parser.parse_args()
config = vars(args)
print(config)

#################
## GLOBAL OPTIONS
#################

# Choose whether the program runs with the live camera (0) or static images (1)
cam_port = 0
if args.extcam: cam_port = 1
# Default image path, just before train/test/coin
IMGPATH = args.imglocation

##initialisation
global img_save_path
global coin_save_path
global valContrast

N = 2
mean_container = ['i',0,0,0]*N
std_container = ['i',0,0,0]*N
min_container = ['i',0,0,0]*N
max_container = ['i',0,0,0]*N
color_mean_lr=np.zeros(3)
HSVmu=[]
feature_space_mean = []
feature_space = []
meancalcvector=[]
new_feature_values_test=[]
cards_feature_list=[]
train_shapes_filename=[]
train_shapes_array = []
train_shapes_number=[]
train_shapes_avg_calc=[]
feature_space_std_train = []
feature_space_mean_train = []
train_shapes_number=[]
feature_space = []
meancalcvector=[]
test_card_feature_identified=''
PN = 1
feature_container = ['i']*PN

##Settings
img_counter = 0
valContrast = 1.06
valBrightness = 11.61
valTresh = 174
valCannyUp=200
valCannyLow=100
sizetopixel=0.1771245765686035
img_save_path = args.imglocation +'/Train'
coin_save_path= args.imglocation +'/Coin'
color_range = ['Yellow','Blue','Green','Red']
choice_sample_all= 'NO'
index=0
features_dict = ['axes_ratio','concavity_ratio','convexity_ratio','area_ratio','vertex_approx','length','perimeter_ratio']
use_features = [0, 1, 2, 3, 4, 5, 6] # 0..6
feature_list = [features_dict[ft] for ft in use_features]

### This local function will list all files contained into the train directory and return the file list and the max index card.
### This function is used when adding cards to the test directory.
def testdir_file_list (img_save_path):
    files = glob.glob (img_save_path + "/*.PNG")
    card_max_num=0
    MyFileList=[]
    for idx, myFile in enumerate(files):
        MFL=len(myFile)
        MyFilePosition= int(myFile.find('\\', len(img_save_path)))+1
        MyFileLen=int(len(myFile)-MyFilePosition)
        MyFileName=str(myFile)[MyFilePosition:]
        MyFileList.append(MyFileName)
        MyFileName=str(MyFileName)[:MyFileLen-4]
        MyFilePosition= int(MyFileName.find('_card', 0)) #also, length of color
        ColorFile=MyFileName[:MyFilePosition-4]
        NumberFile=MyFileName[MyFilePosition-4+10:]
        NumberCard=int(MyFileName[MyFilePosition-3:MyFilePosition])
        cards_feature_list.append([ColorFile,NumberCard,NumberFile])
        MFL_dict= list(dict.fromkeys(MyFileList))
        if NumberCard>card_max_num:card_max_num=NumberCard
    return cards_feature_list, card_max_num, MFL_dict

### This function can be chosen by the user to set the camera distance, hence the conversion of size to pixel.
### This helps to maintain the feature recognition correct, by applying the same scaling no matter the distance from the camera.
def dropthecoin (coin_save_path):
    if args.setsize: #This will execute the dimensioning of the work space, given the position of the camera
        print('Please drop your coin in front of the camera. \n'+
            'Once the green blob is clearly identifying the coin, press the "p" key. ')
        #these are the parameters for the blob detection.
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True # Set Area filtering parameters
        params.minArea = 300
        params.filterByCircularity = True # Set Circularity filtering parameters
        params.minCircularity = 0.85
        params.filterByConvexity = True # Set Convexity filtering parameters
        params.minConvexity = 0.2
        params.filterByInertia = True # Set inertia filtering parameters
        params.minInertiaRatio = 0.01
        
        # Create the trackbar window
        cv2.namedWindow("Treshold ToolBox")
        cv2.createTrackbar("Contrast", "Treshold ToolBox", 0, 25500, updateValContrast)
        cv2.createTrackbar("Brightness", "Treshold ToolBox", 0, 25500, updateValBrightness)
        cv2.createTrackbar("Treshold", "Treshold ToolBox", 0, 255, updateValTresh)
        cv2.createTrackbar("Canny Up", "Treshold ToolBox", 0, 255, updateCannyUp)
        cv2.createTrackbar("Canny Low", "Treshold ToolBox", 0, 255, updateCannyLow)
        cap = cv2.VideoCapture(cam_port)
        img_counter=0
        while(True):
            ret, frame = cap.read()  # Capture frame-by-frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame[:,:,2] = np.clip(valContrast * frame[:,:,2] + valBrightness, 0, 255)
            frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
            #img_colour = cv2.imread(coin_save_path)   # now I do not need it as I am feeding from frame
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # convert to B/W
            img_sm = cv2.blur(img, (5, 5))         # smoothing
            thr_value, img_th = cv2.threshold(img_sm, valTresh, 255, cv2.THRESH_BINARY)   # binarisation
            kernel = np.ones((5, 5), np.uint8)
            #img_th=cv2.erode(img_th,kernel)
            img_close = cv2.morphologyEx(img_th, cv2.MORPH_OPEN, kernel)      # morphology correction
            img_canny = cv2.Canny(img_close, valCannyLow, valCannyUp)                          # edge detection
            contours, hierarchy = cv2.findContours(img_close, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)   # extract contours on binarised image, not on canny
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)         # paint contours on top of original coloured mage
            cv2.imshow('img_close', img_close)
            #cv2.imshow('original with drawn countours', frame)
            #cv2.imshow('picture1', img_th)
            #cv2.imshow('contours', img_canny)

            # Blob routine; I need the blob to adjust the size-to-pizel ratio
            detector = cv2.SimpleBlobDetector_create(params) # Create a detector with the parameters
            keypoints = detector.detect(img_canny) # Detect blobs
            blank = np.zeros((1, 1)) 
            blobs = cv2.drawKeypoints(img_canny, keypoints, blank, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # Draw blobs on our image as red circles
            number_of_blobs = len(keypoints)
            text = "Number of Circular Blobs: " + str(len(keypoints))
            cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
            #print(text)
            #print(keypoints)
            cv2.imshow("Filtering Circular Blobs Only", blobs)
            
            # Display the resulting frame within the brightness regulation window
            cv2.imshow("Treshold ToolBox", frame)
            k = cv2.waitKey(1)
            if k & 0xFF == ord('q'): # Quit key pressed
                print("Escape hit, closing...")
                cap.release()
                cv2.destroyAllWindows()
                break
            elif k & 0xFF == ord('p'): # Photo key pressed
                img_name = "coin_{}.png".format(img_counter)
                cv2.imwrite(coin_save_path + '/' + img_name, frame)
                print("{} written! To continue, close the picture. ".format(img_name))
                img_counter += 1
                break
        coin_save_path= coin_save_path  + '/' + img_name
        # When everything is done, release the capture
        cap.release()
        cv2.destroyAllWindows()

        #Show the blob and calculate the size to pixel
        for keyPoint in keypoints:
            pos_x = keyPoint.pt[0]
            pos_y = keyPoint.pt[1]
            diameter_kp = keyPoint.size
            print('X= ' + str(pos_x) +', Y= ' + str(pos_y) + ', Diam= ' + str(diameter_kp))
        print('To continue, close Figure 1. ')
        plt.imshow(blobs)
        plt.show()
        sizetopixel = diameter_kp/200 #expressed in centimeters, given by the size of the coin
        print(sizetopixel)        
    return sizetopixel

### Routine regulating the camera setup, brightness and contrast, by using tuning trackbars. ******
def updateValContrast(data):
    global valContrast
    valContrast = data/100
    #print("Val Contrast: ", valContrast)
def updateValBrightness(data):
    global valBrightness
    valBrightness = data/100
    #print("Val Brightness: ", valBrightness)
def updateValTresh(data):
    global valTresh
    valTresh = data
    #print("Val treshold: ", valTresh)
def updateCannyUp(data):
    global valCannyUp
    valCannyUp = data
    #print("Val Canny Up: ", valCannyUp)
def updateCannyLow(data):
    global valCannyLow
    valCannyLow = data
    #print("Val Canny Low: ", valCannyLow)
    
### This function will plot the gaussian curves for each color. so the user can see if there is an accaptable distance between colors
def plot_gaussian_graph(mean_container):
    colours = {}
    if mean_container[0][0]=='Yellow': colours[mean_container[0][0]] = [mean_container[0][1], std_container[0][1]] #Yellow
    if mean_container[1][0]=='Blue': colours[mean_container[1][0]] = [mean_container[1][1], std_container[1][1]] #Blue
    if mean_container[2][0]=='Green': colours[mean_container[2][0]] = [mean_container[2][1], std_container[2][1]] #Green
    if mean_container[3][0]=='Light Red': colours[mean_container[3][0]] = [mean_container[3][1], std_container[2][1]] #Light Red
    if mean_container[4][0]=='Dark Red': colours[mean_container[4][0]] = [mean_container[4][1], std_container[4][1]] #Dark Red
    plot_colour = {mean_container[0][0]:'y', mean_container[1][0]:'b', mean_container[2][0]:'g', 
                mean_container[3][0]:'r', mean_container[4][0]:'d'}
    # single channel plot
    channel = 0
    for c in colours.keys():
        mu = colours[c][0][channel]
        sigma = colours[c][1][channel]
        min_x = mu - 4*sigma
        max_x = mu + 4*sigma
        x = np.linspace(min_x,max_x,100)
        y = gauss(x,mu,sigma)
        plt.plot(x,y,plot_colour[c])
    plt.legend(colours.keys())
    plt.show()
    # double channel plot
    channels = (0, 2)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.linspace(0,250,500)
    Y = np.linspace(0,250,500)
    X, Y = np.meshgrid(X, Y)
    Z = 0.0*X+0.0*Y
    for c in colours.keys():
        x_mu = colours[c][0][channels[0]]
        x_sigma = colours[c][1][channels[0]]
        y_mu = colours[c][0][channels[1]]
        y_sigma = colours[c][1][channels[1]]
        Z += gauss(X,x_mu,x_sigma)*gauss(Y,y_mu,y_sigma)
        ax.plot_surface(X,Y,Z,cmap=cm.coolwarm, linewidth=0, antialiased=False) #,plot_colour[c])
    plt.show()

 
###  I kept the colour recognition as one separate function, so I can call it as needed, without repeating the code between testing and training of images' colors.
###  Also, in this way I make sure that color training and testing approach is coherent
def identifyHSV_Avg_Std(card_color_choice,img_save_path):
    files = [x for x in os.listdir(img_save_path) if x.startswith(card_color_choice)]
    print(files)
    trainingfilesfound_counter = len(files)
    print('For ' + card_color_choice +', I found ' + str(trainingfilesfound_counter) +' files \n' )
    color = np.zeros(3)
    for i in range(trainingfilesfound_counter):
        filename =os.path.join(img_save_path,files[i])
        #print(filename)
        imgfull = cv2.imread(filename)
        h_full, w_full, c_full = imgfull.shape
        border=int(round(0.088*w_full,0))
        img= imgfull[border:h_full-border, border:w_full-border] #I am cropping the borders as they are white and they add deviation
        h_im, w_im, c_im = img.shape
        #print('width:  ', w_full)
        #print('height: ', h_full)
        #print('channel:', c_full)
        #plt.imshow(imgfull) 
        #print('width:  ', w_im)
        #print('height: ', h_im)
        #plt.imshow(img)
        
    ###Correct way to run a mask, however I decided not to use it, as it reduces accuracy between yellow and light red
        #maskcolor = cv2.resize(maskcolor, (img.shape[1],img.shape[0]))
        #image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img_sm = cv2.blur(image_gray, (3, 3))         # smoothing, before 5,5,
        #thr_value, img_th = cv2.threshold(img_sm,0,255,cv2.THRESH_OTSU)   # binarisation -110        
        #kernel = np.ones((3, 3), np.uint8) #made kernel smaller, before 5,5 
        #img_th=cv2.erode(img_th,kernel)
        #ret_mask, img_th_inv_mask=cv2.threshold(img_th, 0, 255, cv2.THRESH_BINARY_INV)
        #newmask=cv2.bitwise_and(maskcolor,maskcolor,mask=img_th_inv_mask)
        #newcard=cv2.bitwise_or(img,newmask,mask=None)
        #newcard=cv2.bitwise_and(maskcolor,newcard,mask=None)
        #newcard== cv2.blur(newcard, (5, 5))
    ### End of correct masking    
        border_lay= int(round(w_im*0.1,0))
        #https://theailearner.com/2019/03/26/image-overlays-using-bitwise-operations-opencv-python/
        #covering the central part of the picture to get a better average
        layercover1=img[border_lay*2:border_lay*4, border_lay*5:border_lay*9]   #I want to mask parts of the central number, 
                                                                                #using sections of the coloured area of the card img[y:y+height, x:x+width]
        layercover2=img[border_lay*10:border_lay*12, border_lay*0:border_lay*4]              #I want to mask parts of the central number, using sections of the coloured 
                                                                                # area of the card
        rows,cols,channels = layercover1.shape
        roi = layercover1[0:rows, 0:cols ] 
        img2gray = cv2.cvtColor(layercover1,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, valTresh, 255, cv2.THRESH_BINARY_INV)
        mask_inv = cv2.bitwise_not(mask)
        img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        img2_fg = cv2.bitwise_and(layercover1,layercover1,mask = mask)
        out_img = cv2.add(img1_bg,img2_fg)
        img[border_lay*4:rows+(border_lay*4), border_lay*3:cols+border_lay*3] = out_img  #here is where I want to put it into the picture       
        rows,cols,channels = layercover2.shape
        roi = layercover2[0:rows, 0:cols ] 
        img2gray = cv2.cvtColor(layercover2,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, valTresh, 255, cv2.THRESH_BINARY_INV)
        mask_inv = cv2.bitwise_not(mask)
        img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        img2_fg = cv2.bitwise_and(layercover2,layercover2,mask = mask)
        out_img_2 = cv2.add(img1_bg,img2_fg)
        img[border_lay*8:rows+(border_lay*8), border_lay*3:cols+(border_lay*3) ] = out_img_2  #here is where I want to put it into the picture
        img[border_lay*6:rows+(border_lay*6), border_lay*3:cols+(border_lay*3) ] = out_img_2  #here is where I want to put it into the picture
       #cv2.imshow('layer2',layercover2)
       #cv2.imshow('layer1',layercover1)
        #Now I can calculate the color
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, img_th = cv2.threshold(img_gray,0,255, cv2.THRESH_OTSU) #cv2.THRESH_TRUNC +
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,5)) #From 11 to 35 Here I just want all the blue I can get
        img_ero = cv2.dilate(img_th, kernel)
        ret, img_th_inv = cv2.threshold(img_ero,0,255, cv2.THRESH_BINARY_INV) #I need to invert the white with the black
        internal_chain, hierarchy = cv2.findContours(img_th_inv,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)     
        #print(hierarchy)
        #print('no. Of contours =' + str(len(internal_chain)))
        chain_ar = internal_chain[0]
        #print(chain_ar)
        imHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV);   
        HSV = np.zeros(3)
        #print('Chain Ar shape: ' + str(chain_ar.shape[0]))
        imHSV1 = imHSV.copy()
        cv2.drawContours(imHSV1, internal_chain, -1, (0, 255, 0), 1)
        for c in range(chain_ar.shape[0]):
            #print(chain_ar[c][0][1], chain_ar[c][0][0])
            pHSV = imHSV[chain_ar[c][0][1]][chain_ar[c][0][0]];
            #print('This is phsv' + files[i] + str(pHSV))
            #if pHSV[0] != 0:
            HSV = np.vstack((HSV,pHSV))
            HSV = HSV[1:len(HSV)]
            #print(HSV)
        for i, c in enumerate(internal_chain):            # loop through all the found contours   
            #print(internal_chain  )
            cv2.putText(imHSV1, str(i), (c[0, 0, 0]+0, c[0, 0, 1]-10), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))    # identify contour c
            [x,y,w,h] = cv2.boundingRect(c)
            #print(x,y,w,h)
            cv2.rectangle(imHSV1, (x,y), (x+w,y+h), (255, 0, 0), 2)
        if args.debugmode==True: 
            #DIAGNOSTIC**** Activate this to check all different phases of morphological transformation
            cv2.imshow('imagefull',imgfull)
            cv2.imshow('image',img)
            cv2.imshow('thresh',img_th)
            cv2.imshow('eroded',img_ero)
            cv2.imshow('thresh_inv',img_th_inv)
            cv2.imshow('Card in HSV, contours identified',imHSV1)
            print('Shape of ar chain is:' + str(np.shape(chain_ar)))
        HSVmu = np.mean(HSV,0)
        HSVsd = np.std(HSV,0)
        # here I am filling up the array color with one array for each card this stacks the results in the array
        color = np.vstack((color,HSVmu))
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    #this eliminates the [0,0,0] array created when the array was first initialised.    
    color = color[1:len(color)]
    #I need to introduce a check for the red, as it sits between 170 and 180 and between 0 and 10, 
    #I need to split the red in two peaks
    color_array_shape = color.shape
    #print(color_array_shape[0])
    LightRed_array = np.zeros(3)
    DarkRed_array  = np.zeros(3) 
    Anycolor_array = np.zeros(3)
    for color_check_counter in range(len(color)): 
        if 0<=color[color_check_counter][0]<=15:
            #print('light ' +str(color_check_counter))
            #print('extracted:' + str(color[color_check_counter]))
            LightRed_array=np.vstack((LightRed_array,color[color_check_counter]))
        elif 150<=color[color_check_counter][0]<=180:
            #print('dark '+str(color_check_counter))
            #print('extracted'+ str(color[color_check_counter]))
            DarkRed_array=np.vstack((DarkRed_array,color[color_check_counter]))
        else:
            Anycolor_array= color[0:len(color)]
    color_mean_ac = 0.
    color_std_ac = 0.
    color_min_ac = 0.
    color_max_ac = 0.
    color_mean_lr = 0.
    color_std_lr = 0.
    color_min_lr = 0.
    color_max_lr = 0.
    color_mean_lr = 0.
    color_std_lr = 0.
    color_min_lr = 0.
    color_max_lr = 0.   
    DarkRed_array = DarkRed_array[1:len(DarkRed_array)]
    LightRed_array = LightRed_array[1:len(LightRed_array)]
    #print('This is Light Red: '+ str(LightRed_array))   
    #print('This is Dark Red: '+ str(DarkRed_array))
    #print('This is Any other color: '+ str(Anycolor_array))
    #print('For any color other than red:')
    color_mean_ac = np.mean(Anycolor_array,0)
    color_std_ac = np.std(Anycolor_array,0)
    color_min_ac = np.min(Anycolor_array,0)
    color_max_ac = np.max(Anycolor_array,0)
    print('Color samples: ' + str(Anycolor_array))
    #print('Color mean ac: ' + str(color_mean_ac))
    #print('Color std dev ac: ' + str(color_std_ac))
    #print('Color min ac: ' + str(color_min_ac))
    #print('Color max ac: ' + str(color_max_ac))
    #print('For Light red:')
    color_mean_lr = np.mean(LightRed_array,0)
    color_std_lr = np.std(LightRed_array,0)
    color_min_lr = np.min(LightRed_array,0)
    color_max_lr = np.max(LightRed_array,0)
    if card_color_choice =='Red': print('Light red samples: ' + str(LightRed_array))
    #print('Light red mean lr: ' + str(color_mean_lr))
    #print('Light red std dev lr: ' + str(color_std_lr))
    #print('Light red min lr: ' + str(color_min_lr))
    #print('Light red max lr: ' + str(color_max_lr))
    ##print('For Dark red:')
    color_mean_dr = np.mean(DarkRed_array,0)
    color_std_dr = np.std(DarkRed_array,0)
    color_min_dr = np.min(DarkRed_array,0)
    color_max_dr = np.max(DarkRed_array,0)    
    if card_color_choice =='Red': print('Dark red samples dr: ' + str(DarkRed_array))
    #print('Dark red mean dr dr: ' + str(color_mean_dr))
    #print('Dark red std dev dr: ' + str(color_std_dr))
    #print('Dark red min dr: ' + str(color_min_dr))
    #print('Dark red max dr: ' + str(color_max_dr))
    color_mean = np.mean(color,0)
    color_std = np.std(color,0)
    color_min = np.min(color,0)
    color_max = np.max(color,0)
    #print('This is the final color' + str(color))
    #print('Color mean is: ' + str(color_mean))
    #print('Color std dev is: ' + str(color_std))
    #print('Color min is: ' + str(color_min))
    #print('Color max is: ' + str(color_max))
    mask = cv2.inRange(imHSV, (0, 0, 0), (255, 255,255))
    imginvented=imHSV.copy()
    imginvented[mask > 0]= color_mean #color_max
    imginventedBGR=cv2.cvtColor(imginvented, cv2.COLOR_HSV2BGR)
    if args.debugmode==True: 
        #DIAGNOSTIC***Show final color obtained in BGR, after masking and extracting HSV    
        cv2.imshow('Image BGR',imginventedBGR)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    #f, axarr = plt.subplots(1,2) 
    # use the created array to output your multiple images. In this case I have stacked 2 images vertically
    #axarr[0].imshow(cv2.cvtColor(imHSV, cv2.COLOR_HSV2RGB))
    #axarr[0].imshow(cv2.cvtColor(imgfull, cv2.COLOR_BGR2RGB))
    #plt.show()

    return color_mean, color_std, color_mean_lr, color_std_lr,color_min_lr, color_max_lr, color_mean_dr, color_std_dr,color_min_dr, color_max_dr, color_mean_ac, color_std_ac, color_min_ac, color_max_ac;
    
#https://www.lifewire.com/what-is-hsv-in-design-1078068
#https://www.bing.com/images/search?view=detailV2&ccid=sgpz1erb&id=51EE5192DA95DF208BE92C2B2E0C2D47908DC3D9&thid=OIP.sgpz1erbZI3UgRLXU0oWrAHaBn&mediaurl=https%3a%2f%2fth.bing.com%2fth%2fid%2fR.b20a73d5eadb648dd48112d7534a16ac%3frik%3d2cONkEctDC4rLA%26riu%3dhttp%253a%252f%252fpostfiles14.naver.net%252fMjAxNjExMjVfMzMg%252fMDAxNDgwMDI5Nzk2Mjcz.1Sf8tA_UCJgxa1wpwisr0ntLQbFlgcsMpPIyBo2S4Eog.0jntXLR-ze_o-cuUKS3cjnVVJShNyid0FvRfqiOaItIg.PNG.hayandoud%252fHueScale.svg.png%253ftype%253dw966%26ehk%3ddoRT%252bEgqh94UNrN1C3C53rFwumTJEcyQ%252bX7AbQ2r1Dc%253d%26risl%3d%26pid%3dImgRaw%26r%3d0&exph=210&expw=966&q=hsv+hue+scale&simid=608037734404394850&FORM=IRPRST&ck=96F1BAE1910633D62BB40886F684E3C9&selectedIndex=0&idpp=overlayview&ajaxhist=0&ajaxserp=0

 
### I kept the shape recognition as one separate function, so I can call it as needed, without repeating the code between testing and training of images' features.
### Also, in this way I make sure that feature training and testing approach is coherent
def feature_recognition (frame_feat_input):
    global center
    global orientation
    global axes_ratio 
    global concavity_ratio, convexity_ratio, area_ratio, vertex_approx, length, perimeter_ratio
    global t_perimeter
    global test_hier
    global t_height, t_width, t_channels,thr_value, idx_train_feature
    global ft
    ft=''
    test_cont_mod=[]
    axes_ratio=0.0
    new_feature_values=[]
    features_dict = ['axes_ratio','concavity_ratio','convexity_ratio','area_ratio','vertex_approx','length','perimeter_ratio']
    use_features = [0, 1, 2, 3, 4, 5, 6] # 0..6
    feature_list = [features_dict[ft] for ft in use_features]
    curvature_chain = []
    cont_ar=[] #added
    # I am starting from Frame, image in BGR with only the white borders cropped
    #First, I convert in Gray scale
    if frame_feat_input.all!=[]: 
        image_gray = cv2.cvtColor(frame_feat_input, cv2.COLOR_BGR2GRAY)
        img_sm = cv2.blur(image_gray, (3, 3))         # smoothing, before 5,5,
        thr_value, img_th = cv2.threshold(img_sm,0,255,cv2.THRESH_OTSU)   # binarisation -110
        kernel = np.ones((3, 3), np.uint8) #made kernel smaller, before 5,5 
        img_th=cv2.erode(img_th,kernel)
        img_close = cv2.morphologyEx(img_th, cv2.MORPH_OPEN, kernel)      # morphology correction
        img_close = cv2.morphologyEx(img_th, cv2.MORPH_CLOSE, kernel)      # morphology correction
        img_canny = cv2.Canny(img_close, valCannyLow, valCannyUp)         # edge detection
        contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)   # extract contours on binarised image, not on canny   
        frame_Clear=frame_feat_input.copy()   
        cv2.drawContours(frame_feat_input, contours, -1, (0, 0, 255), 1)         # paint contours on top of original coloured image
    #cv2.drawContours(frame, contours, -1, (0, 0, 255), 1)         # paint contours on top of original coloured mage
    shape_areamax=0
    target_shape=[2,2,2,2]
    for i, c in enumerate(contours):            # loop through all the found contours
        if hierarchy[0, i][0]==-1 and hierarchy[0, i][3]!=-1:
            #print(i, ':', hierarchy[0, i])          # display contour hierarchy
            #print('length: ', len(c))               # display number of points in contour c
            perimeter = cv2.arcLength(c, True)      # perimeter of contour c (curved length)
            #print('perimeter: ', perimeter)               
            epsilon = 5 # 0.01*perimeter            # parameter of polygon approximation: smaller values provide more vertices
            vertex_approx = len(cv2.approxPolyDP(c, epsilon, True))     # approximate with polygon
            #print('approx corners: ', vertex_approx, '\n')              # number of vertices
            cv2.drawContours(frame_feat_input, [c], 0, (0, 255, 0), 3)   # paint contour c
            card_name =str(i)
            cv2.putText(frame_feat_input, card_name, (c[0, 0, 0]+0, c[0, 0, 1]-10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))    # identify contour c
            [x,y,w,h] = cv2.boundingRect(c)
            #print(x,y,w,h)
            cv2.rectangle(frame_feat_input, (x,y), (x+w,y+h), (255, 0, 0), 2)
            shape_area=w*h
            if shape_area > shape_areamax: 
                shape_areamax=shape_area
                target_shape=[x,y,w,h]
        
    # I want a standard picture of 400 by 400
    X_centre= int(target_shape[0]+round(target_shape[2]/2,0))
    Y_centre= int(target_shape[1]+round(target_shape[3]/2,0))
    Y_radius=int(round(target_shape[3]/2,0))
    X_radius=int(round(target_shape[2]/2,0))
    Y_boundary=int(200-Y_radius) # I want height of 400
    X_boundary=int(200-X_radius) # I want width of 400
    Y_crop=Y_centre-Y_radius-Y_boundary
    Y_height=Y_centre+Y_radius+Y_boundary
    X_crop=X_centre-X_radius-X_boundary
    X_width=X_centre+X_radius+X_boundary
    if X_crop<0: X_crop=0 #sometimes the border coincides with the square identifying the shape, X_crop can never be negative!
    if Y_crop<0:Y_crop=0
    shape_cropped= frame_Clear[Y_crop:Y_height, X_crop:X_width] #I am cropping the borders as they are white and they add deviation [y,x]
    shape_cropped_close= img_close[Y_crop:Y_height, X_crop:X_width] #I am cropping the borders as they are white and they add deviation [y,x]
    #I want to eliminate some up left, down right shapes that are creating unwanted contours. All squares that start from X,Y=0 or X Max, Y Max are eliminated
    test_cont, test_hier = cv2.findContours(shape_cropped_close,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)   # extract contours on binarised image  
    if args.debugmode==True:  
        #****DIAGNOSTIC**** Activate this to check all different phases of morphological transformation
        plt.imshow(shape_cropped) 
        cv2.imshow('Frame',frame)
        cv2.imshow('ShapeCropped',shape_cropped)
        cv2.imshow('FrameClear',frame_Clear)
        #cv2.imshow('Crop close',shape_cropped_close)
        cv2.imshow('Close',img_close)
        cv2.imshow('Image gray features 2',image_gray)
        cv2.imshow('Frame+contours',frame_feat_input)
        if shape_cropped.all: cv2.imshow('ShapeCropped',shape_cropped)
        cv2.imshow('canny',img_canny)
    shape_area=0
    t_perimeter=0
    for i_test, c_test in enumerate(test_cont):            # loop through all the found contours
            t_height, t_width, t_channels = shape_cropped.shape
            [x,y,w,h] = cv2.boundingRect(c_test)
            shape_area=w*h
            if (x>=15 and y>=15) and x+w!=t_width: #here I am excluding the outer borders or any border that is too small
                #print(test_cont[i_test])
                test_cont_mod.append(test_cont[i_test])
                #test_cont[i_test].pop
                #print('eliminated') 
                #print(x,y,w,h)
                t_perimeter = cv2.arcLength(c_test, True)      # perimeter of contour c (curved length)
                #print('perimeter: ', perimeter)               
                epsilon = 5 # 0.01*perimeter            # parameter of polygon approximation: smaller values provide more vertices
                vertex_approx = len(cv2.approxPolyDP(c_test, epsilon, True))     # approximate with polygon
                #print('approx corners: ', vertex_approx, '\n')              # number of vertices
    croppedshapecontour=[]
    if np.size(test_cont_mod)>0:
        ch = 0
        n_obj = len(test_cont_mod)
        if ch < n_obj:
            while ch < n_obj and len(test_cont_mod[ch]) < 10: #ch is the index, n_obj counts the objects, therefore the Max(index) =no of objects-1
                ch += 1
            if ch < n_obj: croppedshapecontour = test_cont_mod[ch]
            #Now I calculate and store the features
            curvature_threshold = 0.08
            k = 4
            polygon_tolerance = 0.04
            chain = croppedshapecontour
            cont_ar = np.asarray(chain)
            # compute axes feature
            if len(cont_ar) !=0: 
                ellipse = cv2.fitEllipse(cont_ar)
                (center,axes,orientation) = ellipse
                majoraxis_length = max(axes)
                minoraxis_length = min(axes)
                axes_ratio = minoraxis_length/majoraxis_length
                area = cv2.contourArea(cont_ar)
                perimeter = cv2.arcLength(cont_ar,True)
                area_ratio = perimeter / area
                perimeter_ratio = minoraxis_length / perimeter 
                epsilon = polygon_tolerance*perimeter
                vertex_approx = 1.0 / len(cv2.approxPolyDP(cont_ar,epsilon,True))
                length = len(croppedshapecontour)
                # compute curvature and convexity features
                for i in range(cont_ar.shape[0]-k):
                    num = cont_ar[i][0][1]-cont_ar[i-k][0][1] # y
                    den = cont_ar[i][0][0]-cont_ar[i-k][0][0] # x
                    angle_prev = -np.arctan2(num,den)/np.pi
                    num = cont_ar[i+k][0][1]-cont_ar[i][0][1] # y
                    den = cont_ar[i+k][0][0]-cont_ar[i][0][0] # x
                    angle_next = -np.arctan2(num,den)/np.pi
                    new_curvature = angle_next-angle_prev
                    curvature_chain.append(new_curvature)
                convexity = 0
                concavity = 0
                for i in range(len(curvature_chain)):
                    if curvature_chain[i] > curvature_threshold:
                        convexity += 1
                    if curvature_chain[i] < -curvature_threshold:
                        concavity += 1     
                convexity_ratio = convexity / float(i+1)
                concavity_ratio = concavity / float(i+1)
            
            #here I get the features for each cropped image
            new_feature_values = [eval(ft) for ft in feature_list]
    
    return new_feature_values, NumberFile

# This function plots gaussians to identify each colour 
def gauss(x, mu, sigma):
    return (2*np.pi*sigma**2)**(-.5) * np.exp(-(x-mu)**2/(2*sigma**2))



#*   
#**
#***
#****
#***** this is the script to recognise the color testing of the previous training
def recognisecard(frameinput, valContrast, valBrightness, valTresh, valCannyUp, valCannyLow):
    x=0
    y=0
    w=0
    h=0
    test_frame=[]
    chain_ar=[]
    HSVmu=[]
    test_card_colour_identified =''
    test_card_feature_identified=''
    
    colours = {}
    if mean_container[0][0]=='Yellow': colours[mean_container[0][0]] = [mean_container[0][1], std_container[0][1]] #Yellow
    if mean_container[1][0]=='Blue': colours[mean_container[1][0]] = [mean_container[1][1], std_container[1][1]] #Blue
    if mean_container[2][0]=='Green': colours[mean_container[2][0]] = [mean_container[2][1], std_container[2][1]] #Green
    if mean_container[3][0]=='Light Red': colours[mean_container[3][0]] = [mean_container[3][1], std_container[2][1]] #Light Red
    if mean_container[4][0]=='Dark Red': colours[mean_container[4][0]] = [mean_container[4][1], std_container[4][1]] #Dark Red

    test_frame = cv2.cvtColor(frameinput, cv2.COLOR_BGR2HSV)
    if valContrast==0: valContrast=50
    test_frame[:,:,2] = np.clip(valContrast * test_frame[:,:,2] + valBrightness, 0, 255)
    test_img_full = cv2.cvtColor(test_frame, cv2.COLOR_HSV2BGR)
    frame = cv2.cvtColor(test_img_full, cv2.COLOR_BGR2HSV)
    # Capture frame-by-frame
    frame[:,:,2] = np.clip(valContrast * frame[:,:,2] + valBrightness, 0, 255)
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # convert to B/W
    img_sm = cv2.blur(img, (3, 3))         # smoothing, before 5,5,  
    thr_value, img_th = cv2.threshold(img_sm, valTresh, 255, cv2.THRESH_BINARY)   # binarisation -110
    kernel = np.ones((3, 3), np.uint8) #made kernel smaller, before 5,5 
    img_th=cv2.erode(img_th,kernel)    
    img_close = cv2.morphologyEx(img_th, cv2.MORPH_OPEN, kernel)      # morphology correction
    img_close = cv2.morphologyEx(img_th, cv2.MORPH_CLOSE, kernel)      # morphology correction
    img_canny = cv2.Canny(img_close, valCannyLow, valCannyUp)         # edge detection

    contours, hierarchy = cv2.findContours(img_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)   # extract contours on binarised image, not on canny
    frame_Clear=frame.copy()
    cv2.drawContours(frame, contours, -1, (0, 0, 255), 1)         # paint contours on top of original coloured image
    maxArea = 0.0
    for indexMaxArea, c in enumerate(contours):            # loop through all the found contours
        area = cv2.contourArea(contours[indexMaxArea])
        if area > maxArea:
            maxArea = area
    for i, c in enumerate(contours):            # loop through all the found contours
        if cv2.contourArea(contours[i])==maxArea:
            perimeter = cv2.arcLength(c, True)      # perimeter of contour c (curved length)
            epsilon = 5 # 0.01*perimeter            # parameter of polygon approximation: smaller values provide more vertices
            vertex_approx = len(cv2.approxPolyDP(c, epsilon, True))     # approximate with polygon
            #print('approx corners: ', vertex_approx, '\n')              # number of vertices
            #cv2.drawContours(frame, [c], 0, (0, 255, 0), 3)   # paint contour c
            #card_name = "Card" + str(i)
            #cv2.putText(frame, card_name, (c[0, 0, 0]+0, c[0, 0, 1]-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))    # identify contour c
            [x,y,w,h] = cv2.boundingRect(c)
            #print(contours)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)         
    #print(x,y,w,h)
    cropped_card=frame_Clear[y:y+h , x:x+w]
    [cropcard_h,cropcard_w, channels] = cropped_card.shape
    size_multiplier= int(round(1/sizetopixel,0))
    up_points = (cropcard_w*size_multiplier, cropcard_h*size_multiplier)
    if cropped_card.size!=0 and args.testclass==False: 
        resized_up = cv2.resize(cropped_card, up_points, interpolation= cv2.INTER_LINEAR)
    else:
        resized_up=cropped_card
    #cv2.imshow('a',resized_up) #QUICK CHECK HERE! Make sure the frame is populated until the end!!
    
    h_rs, w_rs, c_rs = resized_up.shape
    if args.debugmode==True: 
        #DIAGNOSTIC**** resized shape
        print('width:  ', w_rs)
        print('height: ', h_rs)
        print('channel:', c_rs)
    border=int(round(0.088*w_rs,0))   
    test_img_cropped= resized_up[border:h_rs-border, border:w_rs-border] #I am cropping the borders as they are white and they add deviation
    test_img_cropped_masked=test_img_cropped.copy()
    h_im, w_im, c_im = test_img_cropped.shape
    
    
    #cv2.imshow("Cropped Card", cropped_card)    
    #print('width:  ', w_im)
    #print('height: ', h_im)
    #plt.imshow(img)
    border_lay= int(round(w_im*0.1,0))
    #https://theailearner.com/2019/03/26/image-overlays-using-bitwise-operations-opencv-python/
    #covering the central part of the picture to get a better average
    
    layercovert1=test_img_cropped[border_lay*2:border_lay*4, border_lay*5:border_lay*9]     #I want to mask parts of the central number, 
                                                                                                #using sections of the coloured area of the card img[y:y+height, x:x+width]
    layercovert2=test_img_cropped[border_lay*10:border_lay*12, border_lay*0:border_lay*4]   #I want to mask parts of the central number, using sections of the coloured 
                                                                                                # area of the card
    if layercovert1.all:                                                                                  
        rows,cols,channels = layercovert1.shape
        roi = layercovert1[0:rows, 0:cols ]
    #print(layercovert1.shape)
    if layercovert1.size>0:
        img2gray = cv2.cvtColor(layercovert1,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, valTresh, 255, cv2.THRESH_BINARY_INV)
        mask_inv = cv2.bitwise_not(mask)
        img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        img2_fg = cv2.bitwise_and(layercovert1,layercovert1,mask = mask)
        out_img = cv2.add(img1_bg,img2_fg)
        if test_img_cropped_masked[border_lay*4:rows+(border_lay*4), border_lay*3:cols+border_lay*3].any==out_img.any: 
            test_img_cropped_masked[border_lay*4:rows+(border_lay*4), border_lay*3:cols+border_lay*3] = out_img  #here is where I want to put it into the picture
    if layercovert2.all:   
        rows,cols,channels = layercovert2.shape
        roi_2 = layercovert2[0:rows, 0:cols ] 
    if layercovert2.size>0:
        img2gray = cv2.cvtColor(layercovert2,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, valTresh, 255, cv2.THRESH_BINARY_INV)
        mask_inv = cv2.bitwise_not(mask)
        img1_bg = cv2.bitwise_and(roi_2,roi_2,mask = mask_inv)
        img2_fg = cv2.bitwise_and(layercovert2,layercovert2,mask = mask)
        out_img_2 = cv2.add(img1_bg,img2_fg)
        if test_img_cropped_masked[border_lay*8:rows+(border_lay*8), border_lay*3:cols+(border_lay*3) ].any == out_img_2.any:
            test_img_cropped_masked[border_lay*8:rows+(border_lay*8), border_lay*3:cols+(border_lay*3) ] = out_img_2  #here is where I want to put it into the picture
            test_img_cropped_masked[border_lay*6:rows+(border_lay*6), border_lay*3:cols+(border_lay*3) ] = out_img_2  #here is where I want to put it into the picture
    #cv2.imshow('layer2',layercover2)
    #cv2.imshow('layer1',layercover1)
    
    #Now I can calculate the color
    if len(test_img_cropped_masked)!=0:
        test_img_gray = cv2.cvtColor(test_img_cropped_masked,cv2.COLOR_BGR2GRAY)
        ret, test_img_th = cv2.threshold(test_img_gray,0,255, cv2.THRESH_OTSU) #cv2.THRESH_TRUNC +
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,5)) #From 11 to 35 Here I just want all the blue I can get
        test_img_ero = cv2.dilate(test_img_th, kernel)
        ret, test_img_th_inv = cv2.threshold(test_img_ero,0,255, cv2.THRESH_BINARY_INV) #I need to invert the white with the black
        internal_chain, hierarchy = cv2.findContours(test_img_th_inv,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)     
        if len(internal_chain)!=0: chain_ar = internal_chain[0]
        imHSV = cv2.cvtColor(test_img_cropped_masked,cv2.COLOR_BGR2HSV)
        HSV = np.zeros(3)
        for c in range(chain_ar.shape[0]):
            pHSV = imHSV[chain_ar[c][0][1]][chain_ar[c][0][0]];
            if pHSV[0] != 0:
                HSV = np.vstack((HSV,pHSV))
        HSV = HSV[1:len(HSV)]
        HSVmu = np.mean(HSV,0)
        HSVsd = np.std(HSV,0)
    #print('The HSV mu value is: ' + str(HSVmu)) 
    if args.debugmode==True: 
        #DIAGNOSTIC*****Here I can see each morphological transformation
        cv2.imshow('thresh',test_img_th)
        cv2.imshow('eroded',test_img_ero)
        cv2.imshow('thresh_inv',test_img_th_inv)
        cv2.imshow("Resized, cropped Card", resized_up)    
        if test_img_cropped.all: cv2.imshow("Cropped Card w/o borders", test_img_cropped)
        if test_img_cropped_masked.all: cv2.imshow("Img cropped masked", test_img_cropped_masked)   
        print (HSVmu.size)
    # closeness is akin to the inverse of probability, lower values correspond to higher probabilities
    
    if HSVmu.size!=1 and HSVmu.all!=[]:  #when it does not get the picture, the size of HSVmu falls from 3 to 1. this if should avoid errors
        closeness = {}
        # summing the distances of H, S, V from the averages found for each colour, divided by the standard deviations
        #print(colours)
        for col in colours.keys():
            closeness1=abs(HSVmu[0] - colours[col][0][0]) / colours[col][1][0]
            closeness2=abs(HSVmu[1] - colours[col][0][1]) / colours[col][1][1]
            closeness3=abs(HSVmu[2] - colours[col][0][2]) / colours[col][1][2]
            closeness[col] = (abs(HSVmu[0] - colours[col][0][0]) / colours[col][1][0])\
                            + abs(HSVmu[1] - colours[col][0][1]) / colours[col][1][1]\
                            + abs(HSVmu[2] - colours[col][0][2]) / colours[col][1][2]   # V
            #print (colours[col][1][0],colours[col][1][1], colours[col][1][2] )
        
        # I create two lists, sorted by values. As they are both sorted the same way and can only contain 5 elements, 
        # I can then use the min of v as an index to derive the color this minimal difference corresponds to.
        ks = list(closeness.keys())
        vs = list(closeness.values())
        i=0
        k=0
        for v in sorted(vs):
            k = ks[vs.index(v)]
            if i==0:
                test_card_colour_identified= k
            #print(k, ' : ', v)
            i=i+1
        #print('Color: ' + test_card_colour_identified)
    if args.debugmode==True: 
        #DIAGNOSTIC****Plot all morphological transformations, in case I need to relate them to the size of the picture
        f, axarr = plt.subplots(2,3) 
        # use the created array to output your multiple images. In this case I have stacked 4 images vertically
        axarr[0][0].imshow(cv2.cvtColor(test_img_full, cv2.COLOR_BGR2RGB))
        axarr[0][1].imshow(cv2.cvtColor(test_img_cropped, cv2.COLOR_BGR2RGB))
        axarr[0][2].imshow(cv2.cvtColor(test_img_cropped_masked, cv2.COLOR_BGR2RGB))
        axarr[1][0].imshow(cv2.cvtColor(layercovert1, cv2.COLOR_BGR2RGB))
        axarr[1][1].imshow(cv2.cvtColor(layercovert2, cv2.COLOR_BGR2RGB))
        axarr[1][2].imshow(cv2.cvtColor(imHSV, cv2.COLOR_BGR2RGB))
        plt.show()
    
#----> I insert here function to return the shape features: it is the same that reads the shape from the test cards
    #cv2.imshow('features input',test_img_cropped)
    new_feature_values_test,NumberFile_test  = feature_recognition (test_img_cropped)
    best_closeness=0
    winning_feat_count=0
    if new_feature_values_test!=[]:
        closeness_feat = {}
        for t_feat_count, fsmt in enumerate(feature_space_mean_train):
            closeness_refshape=fsmt[0]
            closenessf1=abs(new_feature_values_test[0] - feature_space_mean_train[t_feat_count][1][0]) / feature_space_std_train[t_feat_count][1][0]
            closenessf2=abs(new_feature_values_test[1] - feature_space_mean_train[t_feat_count][1][1]) / feature_space_std_train[t_feat_count][1][1]
            closenessf3=abs(new_feature_values_test[2] - feature_space_mean_train[t_feat_count][1][2]) / feature_space_std_train[t_feat_count][1][2]
            closenessf4=abs(new_feature_values_test[3] - feature_space_mean_train[t_feat_count][1][3]) / feature_space_std_train[t_feat_count][1][3]
            closenessf5=abs(new_feature_values_test[4] - feature_space_mean_train[t_feat_count][1][4]) / feature_space_std_train[t_feat_count][1][4]
            closenessf6=abs(new_feature_values_test[5] - feature_space_mean_train[t_feat_count][1][5]) / feature_space_std_train[t_feat_count][1][5]
            closenessf7=abs(new_feature_values_test[6] - feature_space_mean_train[t_feat_count][1][6]) / feature_space_std_train[t_feat_count][1][6]
            if math.isnan(closenessf1):closenessf1=0
            if math.isinf(closenessf1):closenessf1=1000
            if math.isnan(closenessf2):closenessf2=0
            if math.isinf(closenessf2):closenessf2=1000
            if math.isnan(closenessf3):closenessf3=0
            if math.isinf(closenessf3):closenessf3=1000
            if math.isnan(closenessf4):closenessf4=0
            if math.isinf(closenessf4):closenessf4=1000
            if math.isnan(closenessf5):closenessf5=0
            if math.isinf(closenessf5):closenessf5=1000
            if math.isnan(closenessf6):closenessf6=0
            if math.isinf(closenessf6):closenessf6=1000
            if math.isnan(closenessf7):closenessf7=0
            if math.isinf(closenessf7):closenessf7=1000
            closeness_feat[t_feat_count] = closeness_refshape,(closenessf1+closenessf2+closenessf3+closenessf4+closenessf5+closenessf6)
            if t_feat_count==0: 
                best_closeness=closeness_feat[t_feat_count][1]
            elif closeness_feat[t_feat_count][1]<best_closeness: 
                        best_closeness=closeness_feat[t_feat_count][1]
                        winning_feat_count=t_feat_count
            test_card_feature_identified=closeness_feat[winning_feat_count][0]
                        
    cv2.drawContours(frame, contours, -1, (0, 0, 255), 1)         # paint contours on top of original coloured image
    for i, c in enumerate(contours):            # loop through all the found contours
        perimeter = cv2.arcLength(c, True)      # perimeter of contour c (curved length)
        epsilon = 5 # 0.01*perimeter            # parameter of polygon approximation: smaller values provide more vertices
        vertex_approx = len(cv2.approxPolyDP(c, epsilon, True))     # approximate with polygon
        #print('approx corners: ', vertex_approx, '\n')              # number of vertices
        cv2.drawContours(frame, [c], 0, (0, 255, 0), 3)   # paint contour c
        card_name = test_card_colour_identified + " " + test_card_feature_identified
        if args.testclass: 
            cv2.putText(frame, card_name, (200, 200), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))    # in case of static picture, the card name should be inside the card
        else:
            cv2.putText(frame, card_name, (c[0, 0, 0]+0, c[0, 0, 1]-10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))    # identify contour c
        [x,y,w,h] = cv2.boundingRect(c)
        #print(x,y,w,h)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)         
    return frame, card_name

#***************** Here the program Starts - Add Uno Cards to Testing Directory ***********************
#******************************************************************************************************
#******The user has the option to capture additional test cards for training******
if args.addcards: #This will import new cards in the testing folder
    img_counter = 0
    card_name="Train Process"
    x=0
    y=0
    h=0
    w=0
    img_name="NoName"
    if args.setsize: sizetopixel=dropthecoin(coin_save_path)
    print('This program will import additional cards. Once you are satisfied with the quality of the image, press the "p" key to save it. Press "q" to quit.\n')
    file_list, file_max, MFL_dict =testdir_file_list(img_save_path)
    file_max=file_max+1 
    # Create the trackbar window
    cv2.namedWindow("Treshold ToolBox")
    cv2.createTrackbar("Contrast", "Treshold ToolBox", 0, 25500, updateValContrast)
    cv2.createTrackbar("Brightness", "Treshold ToolBox", 0, 25500, updateValBrightness)
    cv2.createTrackbar("Treshold", "Treshold ToolBox", 0, 255, updateValTresh)
    cv2.createTrackbar("Canny Up", "Treshold ToolBox", 0, 255, updateCannyUp)
    cv2.createTrackbar("Canny Low", "Treshold ToolBox", 0, 255, updateCannyLow)
    cap = cv2.VideoCapture(cam_port)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if frame is None:
            print ('Error opening image! Exiting program. \n')
            break
        #here I am experimenting to see if I can adapt the camera to different light conditions
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Capture frame-by-frame
        frame[:,:,2] = np.clip(valContrast * frame[:,:,2] + valBrightness, 0, 255)
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)        
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # convert to B/W
        img_sm = cv2.blur(img, (3, 3))         # smoothing, before 5,5,
        thr_value, img_th = cv2.threshold(img_sm, valTresh, 255, cv2.THRESH_BINARY)   # binarisation -110
        kernel = np.ones((3, 3), np.uint8) #made kernel smaller, before 5,5 
        img_th=cv2.erode(img_th,kernel)
        img_close = cv2.morphologyEx(img_th, cv2.MORPH_OPEN, kernel)      # morphology correction
        img_close = cv2.morphologyEx(img_th, cv2.MORPH_CLOSE, kernel)      # morphology correction
        img_canny = cv2.Canny(img_close, valCannyLow, valCannyUp)         # edge detection
        contours, hierarchy = cv2.findContours(img_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)   # extract contours on binarised image, not on canny
        frame_Clear=frame.copy()                                #Create a BGR clean version of Frame
        cv2.drawContours(frame, contours, -1, (0, 0, 255), 1)   # paint contours on top of original coloured image
        #This below finds all contours, however in this case I only need the outer one, so I used RETR_ETERNAL (above).
        #contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)   # extract contours on binarised image, not on canny
        #cv2.drawContours(frame, contours, -1, (0, 0, 255), 1)         # paint contours on top of original coloured mage
        for i, c in enumerate(contours):            # loop through all the found contours
            #print(i, ':', hierarchy[0, i])          # display contour hierarchy
            #print('length: ', len(c))               # display number of points in contour c
            perimeter = cv2.arcLength(c, True)      # perimeter of contour c (curved length)
            #print('perimeter: ', perimeter)               
            epsilon = 5 # 0.01*perimeter            # parameter of polygon approximation: smaller values provide more vertices
            vertex_approx = len(cv2.approxPolyDP(c, epsilon, True))     # approximate with polygon
            #print('approx corners: ', vertex_approx, '\n')              # number of vertices
            cv2.drawContours(frame, [c], 0, (0, 255, 0), 3)   # paint contour c
            card_name == "Card" + str(i)
            cv2.putText(frame, card_name, (c[0, 0, 0]+0, c[0, 0, 1]-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))    # identify contour c
            [x,y,w,h] = cv2.boundingRect(c)
            #print(x,y,w,h)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)         
        #print(x,y,w,h)
        #cropped_card=frame[90:350 , 220:350]
        cropped_card=frame_Clear[y:y+h , x:x+w]
        [cropcard_h,cropcard_w, channels] = cropped_card.shape
        #image resizing, now I resize the cropped image, in line with the correct shape. 
        #Here it calculates the pixel to size script from drop the coin.
        size_multiplier= int(round(1/sizetopixel,0))
        up_points = (cropcard_w*size_multiplier, cropcard_h*size_multiplier)
        resized_up = cv2.resize(cropped_card, up_points, interpolation= cv2.INTER_LINEAR)
        if args.debugmode==True:  
            #DIAGNOSTIC*****Display the result of each step of shape identification
            cv2.imshow('picture', img)
            cv2.imshow('treshold', img_th)
            cv2.imshow("Step 1: B/W applied", img)
            cv2.imshow("Cropped Card", cropped_card)
            cv2.imshow("Resized, cropped Card", resized_up)
            cv2.imshow('img close', img_close)
            print(contours, hierarchy)
        cv2.imshow("Treshold ToolBox", frame)
        cv2.imshow("Step 2: Canny", img_canny)      
        k = cv2.waitKey(1)
        if k & 0xFF == ord('q'): # Quit key pressed
            print("q for Quit key hit, closing. ")
            cap.release()
            cv2.destroyAllWindows()
            break
        elif k & 0xFF == ord('p'): # Photo key pressed
            user_answer_3 = input('Please provide the colour of the card you want to save. \n ' +
            'Choose between: Yellow, Red, Blue, Green or Special: ')
            user_answer_4 = input('Now please provide the number (e.g. 9) or, for special cards the name \n ' +
            'e.g. Skip, TakeTwo, Reverse, SwapCards, Wild, TakeFour, YourChoice: ')
            print('You are about to save: ' + user_answer_3 + ' ' + user_answer_4)
            file_max_str=str(file_max)
            if len(file_max_str)<3:
                if len(file_max_str)==1: file_max_str='00'+file_max_str
                elif len(file_max_str)==2: file_max_str='0'+file_max_str
            file_max=file_max+1
            img_name = user_answer_3 + "_"+file_max_str+"_card_" + user_answer_4 + ".png".format(img_counter)
            cv2.imwrite(img_save_path + '/' + img_name, resized_up)
            print("Image {} saved. ".format(img_name))
            img_counter += 1        
            print('Close the picture to continue, if you want to stop, press "q": \n')
            if args.debugmode==True:  
                #DIAGNOSTIC*****Here I show the cropped card, plotted in its unit of measure
                f, axarr = plt.subplots(2) 
                # use the created array to output your multiple images. In this case I have stacked 2 images horizontally
                axarr[0].imshow(cv2.cvtColor(frameRGB, cv2.COLOR_BGR2RGB))
                axarr[1].imshow(cv2.cvtColor(resized_up, cv2.COLOR_BGR2RGB))
                plt.show()        
    train_save_path= img_save_path + '/' + img_name
    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    exit()

#*********************************** Program to recognise UNO Cards ***********************************
#******************************************************************************************************


### This script classifies colors for the training cards. I  split red into dark and light, given that it sat at the extremes 
### of the Hue scale and calculating a mean between them would have brought meaningless results.
if args.setsize: sizetopixel=dropthecoin(coin_save_path)
file_count = next(os.walk(img_save_path))[2] #dir is your directory path as string
print ('The directory contains: '+ str(len(file_count))+' files.')
print("Now we will train for all the four colours. \n")
for color_range_counter in color_range:
    #print('The colour in counter is: ' + color_range_counter)        
    color_mean, color_std, color_mean_lr, color_std_lr,color_min_lr, color_max_lr, color_mean_dr, color_std_dr, \
    color_min_dr, color_max_dr, color_mean_ac, color_std_ac, color_min_ac, color_max_ac= identifyHSV_Avg_Std(color_range_counter,img_save_path)
    if color_range_counter =='Red':
        a_lr=np.size(color_mean_lr, axis=None)
        a_dr=np.size(color_mean_dr, axis=None)
        print(a_dr,a_lr)
        #if color_mean_lr[0] != 0.:
            #print('Yes')
        if a_lr!=1:
            print(np.size(color_mean_lr, axis=None))
            mean_container[index]= ("Light " + color_range_counter,color_mean_lr)
            min_container[index]=("Light " + color_range_counter,color_min_lr)
            max_container[index]=("Light " + color_range_counter,color_max_lr)
            if color_std_lr[0]==0:
                color_std_lr[0]=1
                color_std_lr[1]=1
                color_std_lr[2]=1
            std_container[index]= ("Light " + color_range_counter,color_std_lr)
            index=index+1
        if a_dr!=1:
            mean_container[index]= ("Dark " + color_range_counter,color_mean_dr)
            min_container[index]=("Dark " + color_range_counter,color_min_dr)
            max_container[index]=("Dark " + color_range_counter,color_max_dr)    
            if color_std_dr[0]==0:
                color_std_dr[0]=1
                color_std_dr[1]=1
                color_std_dr[2]=1
            std_container[index]= ("Dark " + color_range_counter,color_std_dr)
            index=index+1
    else: 
            mean_container[index]= (color_range_counter,color_mean_ac)
            std_container[index]=  (color_range_counter,color_std_ac)
            min_container[index]=(color_range_counter,color_min_ac)
            max_container[index]=(color_range_counter,color_max_ac)
            index=index+1
print('\n mean: \n' + str( mean_container))
print('\n stdev: \n'+str(std_container))
print('\n min: \n'+str(min_container))
print('\n max: \n'+str(max_container))
if args.plotgaussian: plot_gaussian_graph(mean_container)

### Here I put my shape feature classification function. I do not want to duplicate the pictures in a folder, so I created 
### an array and I go through each one of them to extract their features.
files = glob.glob (img_save_path + "/*.PNG")
for idx, myFile in enumerate(files):
    MFL=len(myFile)
    MyFilePosition= int(myFile.find('\\', len(img_save_path)))+1
    MyFileLen=int(len(myFile)-MyFilePosition)
    MyFileName=str(myFile)[MyFilePosition:]
    MyFileName=str(MyFileName)[:MyFileLen-4]
    MyFilePosition= int(MyFileName.find('_card', 0)) #also, length of color
    ColorFile=MyFileName[:MyFilePosition-4]
    NumberFile=MyFileName[MyFilePosition-4+10:]
    MyFileName= ColorFile +'_'+ NumberFile
    #print(myFile)
    image = cv2.imread (myFile)
    #First of all, I extract the card's central shape, then I start morphing the picture,.
    h_rs, w_rs, c_rs = image.shape
    border=int(round(0.088*w_rs,0))
    shape_cropped= image[border:h_rs-border, border:w_rs-border] #I am cropping the borders as they are white and they add deviation
    h_im, w_im, c_im = shape_cropped.shape
    frameRGB = cv2.cvtColor(shape_cropped, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(shape_cropped, cv2.COLOR_BGR2HSV)
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)  
    frame[:,:,2] = np.clip(valContrast * frame[:,:,2] + valBrightness, 0, 255)   
    #img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # convert to B/W
    # my feature classification defined function is called. It returns the list of shapes and the card number
    new_feature_values,NumberFile = feature_recognition (frame)
    new_feature_values_ar=np.asarray(new_feature_values)
    new_feature_values_T=new_feature_values_ar.transpose()
    feature_container[0]=(NumberFile,new_feature_values_T)
    feature_space.append(feature_container[0])
    train_shapes_number.append (NumberFile)
    feature_space.sort(key=lambda x: x[0]) 
    train_shapes_number_dictionary = list(dict.fromkeys(train_shapes_number))
for ii,idx_train_num in enumerate(train_shapes_number_dictionary):
    meancalcvector=[]
    for iitf,idx_train_feature in enumerate(feature_space):
        if feature_space[iitf][0]==idx_train_num:
            if feature_space[iitf][1]!=[]: meancalcvector.append (feature_space[iitf][1])
    mean_c_features= (idx_train_num,np.mean(meancalcvector, dtype=np.float64, axis=0))
    std_c_features= (idx_train_num,np.std(meancalcvector, dtype=np.float64, axis=0))
    feature_space_mean_train.append(mean_c_features)
    feature_space_std_train.append(std_c_features)
if args.debugmode==True: 
    #****DIAGNOSTIC****
    print(idx_train_num, meancalcvector)    
    print(feature_space_mean_train) #This will print the list of the mean of the 7 dimensions for all card types
    print(feature_space_std_train) #This will print the list of the stNDrd deviation of the 7 dimensions for all card types

### Now we will recognise the cards, based on the features and colors we classified earlier. There will be two options, to 
### be selected by the user, by adding the argument -t, the same pictures used for training can be used for testing.
### leaving this parameter empty will allow for real time comparison.

    # Create the trackbar window
    cv2.namedWindow("Card Finder")
    cv2.createTrackbar("Contrast", "Card Finder", 0, 25500, updateValContrast)
    cv2.createTrackbar("Brightness", "Card Finder", 0, 25500, updateValBrightness)
    cv2.createTrackbar("Treshold", "Card Finder", 0, 255, updateValTresh)
    cv2.createTrackbar("Canny Up", "Card Finder", 0, 255, updateCannyUp)
    cv2.createTrackbar("Canny Low", "Card Finder", 0, 255, updateCannyLow)
k=0
if args.testclass==True:
    continue_test='Y'
    while continue_test=='Y':
        cv2.destroyAllWindows()
        file_list, file_max, MFL_dict =testdir_file_list(img_save_path)
        print('Your available choices are: ' + str(MFL_dict))
        answer_col = input('Which card out of the list above would you like to test? ')
        myFile_test = img_save_path + '//'+ answer_col
        img_test = cv2.imread(myFile_test)
        if img_test.all!=[]:
            frameoutput, cardname= recognisecard(img_test, valContrast, valBrightness, valTresh, valCannyUp, valCannyLow)
        else:
            print("'Wrong argument input, closing. ")
            break
        cv2.imshow("Card Finder", frameoutput)
        print(cardname)
        k = cv2.waitKey(1)
        continue_test = input('do you want to test another card? Y or N: ')
    print("End of program. Bye. ")
    cv2.destroyAllWindows()
    
else: 
    test_cap = cv2.VideoCapture(cam_port)
    print("To exit, press q: \n")
    while(True):
        ret, test_frame = test_cap.read()  # Capture frame-by-frame8
        frameoutput, cardname= recognisecard(test_frame, valContrast, valBrightness, valTresh, valCannyUp, valCannyLow)
        #print(x,y,w,h)
        cv2.imshow("Card Finder", frameoutput)
        k = cv2.waitKey(1)
        if k & 0xFF == ord('q'): # Quit key pressed
                print("q for Quit key hit, closing. ")
                test_cap.release()
                cv2.destroyAllWindows()
                break