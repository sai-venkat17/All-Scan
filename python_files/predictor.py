from flask import Flask,request

app = Flask(__name__)


import cv2
import dropbox
import imutils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import pandas as pd
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
import pickle
import time


    
import pymongo
pca0=pickle.load(open("python_files/pca_model.sav",'rb'))
final_model0=pickle.load(open("python_files/ensemble_model2.sav",'rb'))
pca1=pickle.load(open("python_files/lung_pca_model.sav",'rb'))
final_model1=pickle.load(open("python_files/lung_ensemble_model.sav",'rb'))
mongo_url="mongo_URL"
client=pymongo.MongoClient(mongo_url)
    
def download_file(file_name,local_file_path):
    global client
    db=client['all_scan']
    images=db['images']
    image=images.find_one({"name":file_name})
    
    with open(local_file_path,'wb') as f:
        f.write(image['img']['data'])
        f.close()

def upload_file(file_name,local_file_path):
    global client
    db=client['all_scan']
    images=db['images']
    images.delete_one({"name":file_name})
    with open(local_file_path,'rb') as f:
        image=f.read()
        images.insert_one({"name":file_name,"img":{"data":image,"contentType":"image/jpg"}})
        f.close()



    

def convert_to_grayscale(image):
    gray_scale_img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return gray_scale_img

#Reduces image size and down sampling the image
def gaussian_blur(image):
    return cv2.GaussianBlur(image,(5,5),0)

def threshold_image(image):
    ret,th=cv2.threshold(image,45,255,cv2.THRESH_BINARY)
    return th
def erode_image(image):
     return cv2.erode(image, None, iterations=2)
def dilate_img(image):
    return cv2.dilate(image, None, iterations=2)
    
def crop_image_coordinates(image,plot=False):
    #Grabbing largest contours
    cntrs=cv2.findContours(image.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cntrs = imutils.grab_contours(cntrs)
    largest_cntr = max(cntrs, key=cv2.contourArea)
    #Extreme points finding for cropping
    ext_left = tuple(largest_cntr[largest_cntr[:, :, 0].argmin()][0])
    ext_right = tuple(largest_cntr[largest_cntr[:, :, 0].argmax()][0])
    ext_top = tuple(largest_cntr[largest_cntr[:, :, 1].argmin()][0])
    ext_bot = tuple(largest_cntr[largest_cntr[:, :, 1].argmax()][0])
    return [ext_left,ext_right,ext_top,ext_bot]
def crop_image(image,crop_image_coords):
    return image[crop_image_coords[2][1]:crop_image_coords[3][1], crop_image_coords[0][0]:crop_image_coords[1][0]]
def resize_image(image,size):
    width,height=size
    return cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)

def normalize_image(image):
    return image/255
def prepare_image(image):
    img_list=[]
    gray_scale_img=convert_to_grayscale(image)
    img_list.append(gray_scale_img)
    blur_img=gaussian_blur(gray_scale_img)
    img_list.append(blur_img)
    img_list.append(threshold_image(blur_img))
    img_list.append(erode_image(img_list[-1]))
    img_list.append(dilate_img(img_list[-1]))
    cropped_img=crop_image(image,crop_image_coordinates(img_list[-1]))
    img_list.append(cropped_img)
    img_list.append(normalize_image(resize_image(cropped_img,[240,240])))
    #edges=filters.sobel(image)
    #img_list.append(edges)
    #plt.savefig(edges,"inter/edge_detect.jpg")
    return img_list
    
def predict_image(img_url,img_type):
    uploaded_img=cv2.imread(img_url)
    img_list=prepare_image(uploaded_img)
    input_image=img_list[-1]
    Image=[]
    Image.append(input_image)
    Image=np.array(Image)
    l=[]
    for image_no in range((Image.shape)[0]):
        l.append(Image[image_no].ravel())
    df=pd.DataFrame(l,columns=[i for i in range(len(l[0]))])
    if(img_type==0):
        X_test=pca0.transform(df)
    elif(img_type==1):
        X_test=pca1.transform(df)
    for i in range(len(img_list)-1):
        cv2.imwrite("inter/img"+str(i)+".jpg",img_list[i])
    #cv2.imwrite("inter/edge_detect.jpg",img_list[i])
    if(img_type==0):
        return str(final_model0.predict(X_test)[0])
    elif(img_type==1):
        return str(final_model1.predict(X_test)[0])

@app.route("/")
def hello_world():
    return "Flask server is running at this port"

@app.route("/brain",methods=['POST','GET'])
def predict_brain():
    if(request.method=='POST'):
        download_file(request.form['file_name'],"uploads/"+request.form['file_name'])
        ans=predict_image("uploads/"+request.form['file_name'],0)
        for i in range(6):
            upload_file("img"+str(i),"inter/img"+str(i)+".jpg")
        return ans
    else:
        return "Only post requests are allowed"

@app.route("/lung",methods=['POST','GET'])
def predict_lung():
    if(request.method=='POST'):
        download_file(request.form['file_name'],"uploads/"+request.form['file_name'])
        ans=predict_image("uploads/"+request.form['file_name'],1)
        for i in range(6):
            upload_file("img"+str(i),"inter/img"+str(i)+".jpg")
        return ans
    else:
        return "Only post requests are allowed"
if(__name__=="__main__"):
    app.run(debug=True)
        