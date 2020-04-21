# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 19:03:13 2020

@author: abhis
"""

import numpy as np
import pandas as pd
import cv2
jjdlkfjals = np.zeros((1,1))
import glob
import random
import os
Train_Read = 'TrainData_C2//'
Train_wtite = 'TrainData_PP//'
Test_Read = 'TestData//'
Test_write = 'TestData_PP'
Val_Write = 'ValData_PP//'

def hist_equal(img, flag = False):
    '''takes image and an output space [0,L] as an input and gives an equalized image(in float) as output'''
    if flag:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if len(img.shape) != 2:
        R, G, B = cv2.split(img)
        output1_R = cv2.equalizeHist(R)
        output1_G = cv2.equalizeHist(G)
        output1_B = cv2.equalizeHist(B)
        equ = cv2.merge((output1_R, output1_G, output1_B))
        return equ
    return cv2.equalizeHist(img)


def randomcropper(img):
    '''takes in an image of size 480,480 and gives out an image of 120,120'''
    x = random.randint(0,360)
    y = random.randint(0,360)
    return img[x:x+120,y:y+120,:]
    
def traindata(DataPath_R,DataPath_W,df_R):
    labellist = []
    imagelist = []
    
    for i in df_R.index:
        
        imageName = df_R.at[i,'file_name']
        
        image = cv2.imread(DataPath_R + imageName)
        
        label = df_R.at[i,'annotation']
        
        image = hist_equal(image)
       
        image = cv2.resize(image, (480,480), interpolation = cv2.INTER_AREA)
        for i in range(4):
                for j in range(4):
                    imagepatch = image[120*i:120*(i+1),120*j:120*(j+1),:]
                    print(imagepatch.shape)
                    newImageName = str(i)+str(j)+imageName
                    imagepatch = cv2.resize(imagepatch,(224,224),interpolation = cv2.INTER_AREA)
                    cv2.imwrite(DataPath_W+newImageName,imagepatch)
                    imagelist.append(newImageName)
                    labellist.append(label)
    df_W = pd.DataFrame({'file_name':imagelist,'annotation':labellist})
    return df_W
    
def datasampler(df):
    df_0 = df.loc[df['annotation']==0]
    df_0 = df_0.sample(n=1600).reset_index(drop=True)
    
    
    df_1 = df.loc[df['annotation']==1]
    df_1 = df_1.sample(n=1600).reset_index(drop=True)
    
    df_2 = df.loc[df['annotation']==2]
    df_2 = df_2.sample(n=1600).reset_index(drop=True)
    
    df_3 = df.loc[df['annotation']==3]
    df_3 = df_3.sample(n=1600).reset_index(drop=True)
    
    df_4 = df.loc[df['annotation']==4]
    df_4 = df_4.sample(n=1600).reset_index(drop=True)
    
    finaldf = pd.concat([df_0,df_1,df_2,df_3,df_4])
    
    return finaldf.sample(frac = 1).reset_index(drop=True)
    
    
    
def dataprep(DataPath_R,DataPath_W,df_R,AugmentFlag = True, trainimg = True):
    'Something Goes here'
    labellist = []
    imagelist = []

    for i in df_R.index:
        print(i)
        print(type(i))
        imageName = df_R.at[i,'file_name']
        
        image = cv2.imread(DataPath_R + imageName)
        
        label = df_R.at[i,'annotation']
        
        image = hist_equal(image)
       
        image = cv2.resize(image, (480,480), interpolation = cv2.INTER_AREA)

        if AugmentFlag:
            for i in range(2):
                for j in range(2):
                    imagepatch = image[240*i:240*(i+1),240*j:240*(j+1),:]
                    print(imagepatch.shape)
                    newImageName = str(i)+str(j)+imageName
                    imagepatch = cv2.resize(imagepatch,(224,224),interpolation = cv2.INTER_AREA)
                    if trainimg:
                        if (label == 0):
                            numpick = random.randint(1,3)
                            if (numpick == 1):
                                cv2.imwrite(DataPath_W +'r0'+newImageName,cv2.flip(imagepatch,0))
                                imagelist.append('r0'+newImageName)
                                labellist.append(label)
                        else:
                            imlist ,lblist = dataaugmenter(DataPath_W,newImageName,imagepatch,label)
                            imagelist = imagelist+imlist
                            labellist = labellist+lblist
                    
                    cv2.imwrite(DataPath_W+newImageName,imagepatch)
                    imagelist.append(newImageName)
                    labellist.append(label)
                    
        
        else:
            #image = cv2.resize(image,(224,224),interpolation = cv2.INTER_AREA)
            if trainimg:
                if (label == 0):
                    numpick = random.randint(1,3)
                    if (numpick == 1):
                        cv2.imwrite(DataPath_W +'r0'+imageName,randomcropper(image))
                        imagelist.append('r0'+imageName)
                        labellist.append(label)
                else:
                    imlist ,lblist = dataaugmenter(DataPath_W,imageName,randomcropper(image),label)
                    imagelist = imagelist+imlist
                    labellist = labellist+lblist
            
            #Writing the original Image
            cv2.imwrite(DataPath_W+imageName,randomcropper(image))
            imagelist.append(imageName)
            labellist.append(label)
                    
    df_W = pd.DataFrame({'file_name':imagelist,'annotation':labellist})
    
    return df_W


def dataaugmenter(DataPath_W,imagename,image,label,):
    '''Something goes here'''
    imlist = []
    lblist = []
    cv2.imwrite(DataPath_W + 'hflip0'+imagename,cv2.flip(image,0))
    imlist.append('hflip0'+imagename)
    lblist.append(label)
    
    if(label != 1):
        '''bla bla bla''' 
        cv2.imwrite(DataPath_W+'hflip1'+imagename,cv2.flip(image,1))
        imlist.append('hflip1'+imagename)
        lblist.append(label)
    
    if((label != 4) and (label != 1)):
        cv2.imwrite(DataPath_W+'rdnoice'+imagename,image + 10*np.random.rand(224,224,3))
        imlist.append('rdnoice'+imagename)
        lblist.append(label)
        cv2.imwrite(DataPath_W+'hflip'+imagename,cv2.flip(image,-1))
        imlist.append('hflip'+imagename)
        lblist.append(label)
        
    return imlist,lblist
                   

    

def Testdata(DataPath_R,DataPath_W, AgData = False):
    x = os.getcwd()
    os.chdir(x+'\\'+DataPath_R)
    #print(os.getcwd())
    
    td = glob.glob('*.jpg')
    
    for i in td:
        print(i)
        image = cv2.imread(i)
        image = cv2.resize(image,(480,480),interpolation = cv2.INTER_AREA)
        image = hist_equal(image)
        print(image.shape)
        if AgData:
            '''Something'''
            print('..\\'+DataPath_W+'_1\\'+i)
            cv2.imwrite('..\\'+DataPath_W+'_1\\'+i,image[0:224,0:224,:])
            cv2.imwrite('..\\'+DataPath_W+'_2\\'+i,image[0:224,240:464,:])
            cv2.imwrite('..\\'+DataPath_W+'_3\\'+i,image[240:464,0:224,:])
            cv2.imwrite('..\\'+DataPath_W+'_4\\'+i,image[240:464,240:464,:])
        else:
            itisim = randomcropper(image)
            itisim = cv2.resize(itisim,(244,244),interpolation = cv2.INTER_AREA)
            cv2.imwrite('..\\'+DataPath_W+i,itisim)
    
        
    









def datasplit(df_R):
    '''Something goes here'''
    
    df_R_0 = df_R.loc[df_R['annotation']==0]
    #print(df_R_0)
    
    df_R_0_s = df_R_0.sample(30, random_state = 0)
    #print(df_R_0_s)
    #df_R_0_train = df_R_0.drop(df_R_0_s.file_name)
    
    df_R_1 = df_R.loc[df_R['annotation']==1]
    df_R_1_s = df_R_1.sample(30, random_state = 0)
    #df_R_1_train = df_R_1.drop(df_R_1_s.index)
    
    df_R_2 = df_R.loc[df_R['annotation']==2]
    df_R_2_s = df_R_2.sample(30, random_state = 0)
    #df_R_2_train = df_R_2.drop(df_R_2_s.index)
    
    df_R_3 = df_R.loc[df_R['annotation']==3]
    df_R_3_s = df_R_3.sample(30, random_state = 0)
    #df_R_3_train = df_R_3.drop(df_R_3_s.index)
    
    df_R_4 = df_R.loc[df_R['annotation']==4]
    df_R_4_s = df_R_4.sample(30, random_state = 0)
    #df_R_4_train = df_R_4.drop(df_R_4_s.index)
    
    #train_df = pd.concat([df_R_0_train,df_R_1_train,df_R_2_train,df_R_3_train,df_R_4_train], ignore_index = True)
    #train_df = pd.concat([df_R_0,df_R_1,df_R_2,df_R_3,df_R_4], ignore_index = True)
    val_df = pd.concat([df_R_0_s,df_R_1_s,df_R_2_s,df_R_3_s,df_R_4_s])
    return val_df
       
        

        
    



def main():
    'Something needs to be done here'
    
    # Step1 : Read all the input csv using pandas
    df_TrainOriginal = pd.read_csv(Train_Read+'TrainAnnotations.csv')
    print(df_TrainOriginal.shape)
    
    #Step2: Create validation and training set split (100 images in validation 20 from each class)
    df_val = datasplit(df_TrainOriginal)
    #print(df_T.shape)
    print(df_val.shape)
    traindf = df_TrainOriginal.drop(df_val.index)
    print(traindf.shape)
    print(traindf.head())
    
    
   # Step 3:
    df_valpp = dataprep(Train_Read,Val_Write,df_val,trainimg=False)
    #df_valpp = traindata(Train_Read,Val_Write,df_val)
    print(df_valpp.head())
    df_valpp.to_csv(Val_Write+'Valannotation.csv')
    
    #Step 4:
    df_trainpp = dataprep(Train_Read,Train_wtite,traindf)
    #df_trainpp = traindata(Train_Read,Train_wtite,traindf)
    print(df_trainpp.head())
    print(df_trainpp.shape)
    
    #df_train = df_trainpp.sample(frac = 0.5)
    df_trainpp = datasampler(df_trainpp)
    df_trainpp.to_csv(Train_wtite+'TrainAnnotations.csv')
    
#    Testdata(Test_Read,Test_write)
    
    
if __name__ == '__main__':
    main()