from flask import Flask, render_template, request,flash
from flask import json
import logging
import numpy as np
# import pandas as pd 
import cv2
import csv


# uncomment this after 

import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


STATIC_FOLDER = "./static/"
MODELS_FOLDER = "./models/"


# Load model
cnn_model = load_model(MODELS_FOLDER + "ecg_cnn_model.h5")

app = Flask(__name__,template_folder='templates')
app.secret_key = b'mysecrect'


app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF","jpeg", "jpg", "png", "gif"]

# for allowed images.
def allowed_image(filename):
    if not "." in filename:
        return False
    ext = filename.rsplit(".", 1)[1]
    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False


## Functions for Preprocessing 
# Denoising using Salt-Peper
def salt(img, n):
    for k in range(n):
        i = int(np.random.random() * img.shape[1])
        j = int(np.random.random() * img.shape[0])
        if img.ndim == 2:
            img[j,i] = 255
        elif img.ndim == 3:
            img[j,i,0]= 255
            img[j,i,1]= 255
            img[j,i,2]= 255
        return img


# BG Remover  inspired by : @Messaoud Makhlouf 
def bg_remov(image):
    result = salt(image, 10)
    median = cv2.medianBlur(result,5)
    gray = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(gray, 85, 255, cv2.THRESH_BINARY)
    
    return blackAndWhiteImage


def process(img_path):

    img_array = cv2.imread(img_path)  # read the image from path.
    img_array = bg_remov(img_array)   # remove background
    image_size = 256                  # Image Size
    new_img_array = cv2.resize(img_array, (image_size, image_size))   #Resize the Image.

    return  new_img_array


# Prediction or Classification
def predict_label(model,img_path):
    #i = image.load_img(img_path, target_size=(256,256))
    i = process(img_path)
    i = image.img_to_array(i)/255.0
    i = i.reshape(-1, 256,256,1)
    
    prob = model.predict(i)
    label = "Normal" if prob[0][0] >= 0.5 else "MI"
    classified_prob = prob[0][0] if prob[0][0] >= 0.5 else 1 - prob[0][0]
    
    return label, classified_prob






# Function to Create the CSV file with header.
def init_csvFile():
    # write to a CSV file the real label with the image.
    header = ['img_path', 'img_filename', 'label']
    # open the file in the write mode
    with open(STATIC_FOLDER + 'ecg_true_labels.csv', 'w' ,encoding='UTF8',newline='') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

#init csv outside the classification either way we will create each time 1 raw.
# 
init_csvFile()

# Function to Append new raws to the CSV file.
def append_list_as_row(file_name, list_of_elem):
    
    # Open file in append mode    
    with open(file_name, 'a+', newline='') as f:
        # Create a writer object from csv module
        csv_writer = csv.writer(f)
        
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

# ecg_df = pd.DataFrame( columns = ['FilePath','filename','label'] )



# routes
@app.route("/",methods=['GET','POST'])
def main():

    # log line
    app.logger.info('Welcome Request successfull')

    return render_template('index.html')

# @app.route("/real_inference", methods=['GET','POST'])
# def real_label():
#     if request.method == 'POST':
#         label = 



@app.route("/classification", methods=['GET','POST'])
def heart_risk():
    if request.method == 'GET':
        
        return render_template('classify.html')
    error = None
    if request.method == 'POST' and len(request.files['my_image'].filename) and allowed_image(request.files['my_image'].filename):
        img = request.files['my_image']
        real_label = request.form['doc_infer'] # Real label.
        img_path = STATIC_FOLDER +'images/' + img.filename
        img.save(img_path)

        # model prediction
        label, prob = predict_label(cnn_model,img_path)
        #print(prob[0][0])
        prob = round((prob*100),2)

        data = [img_path, img.filename, real_label]
        
        
        csv_file_name = STATIC_FOLDER+ 'ecg_true_labels.csv'
        # append new raws to csv file.
        append_list_as_row(csv_file_name, data)
        
        return render_template("classify.html", label=label, prob=prob, img_path=img_path, real_label= real_label)
    else: 
        flash('There was a problem uploading that picture, conform files ends with \
                {"jpeg", "jpg", "png", "gif"}')
        return render_template("classify.html")




if __name__ == "__main__":

    ## stream logs to app.log file
    logging.basicConfig(filename='app.log',level=logging.DEBUG)
    
    app.run(debug=True)
