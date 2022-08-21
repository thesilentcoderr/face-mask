import cv2
import streamlit as st 
from PIL import Image
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import img_to_array


st.set_option('deprecation.showfileUploaderEncoding', False)
model = load_model('mask_detection_best.h5')
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Machine Learning/Deep Learning Lab Experiment</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
  
st.title("""
        Face Mask Detection
         """
         )
file= st.file_uploader("Please upload an image", type=("jpg", "png"))

def predict(image):
  face_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  face_frame = cv2.resize(face_frame, (224, 224))
  face_frame = img_to_array(face_frame)
  face_frame = np.expand_dims(face_frame, axis=0)
  face_frame =  preprocess_input(face_frame)
  prediction = model.predict(face_frame)
  
  return prediction[0][0]

def detect(gray_image,frame):
  faces = face_classifier.detectMultiScale(gray_image, 1.1, 5)
  for (x,y,w,h) in faces:
    roi_color = frame[y:y+h, x:x+w]
    mask = predict(roi_color)
  
  if mask > 0.5:
    val = "Mask is ON"
  elif mask<=0.5:
    val = "Mask is OFF"
  return val

if file is None:
  st.text("Please upload an Image file")
else:
  image=Image.open(file)
  image=np.array(image)
  gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  st.image(image,caption='Uploaded Image.', use_column_width=True)
    
if st.button("Predict Expression"):
  result=detect(gray_frame,image)
  st.success('Model has predicted that {}'.format(result))
if st.button("About"):
  st.header("Yash Sankhla")
  st.subheader("Student, Department of Computer Engineering")
  
html_temp = """
   <head>

    <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <script type="text/javascript" src="http://ajax.googleapis.com/ajax/libs/jquery/1.6.2/jquery.min.js"></script>

    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.3.1/dist/css/bootstrap.min.css"
        integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
   </head>
   <body>
  
   <div class="" style="background-color:orange;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:20px;color:white;margin-top:10px;">Deep Learning Experiment</p></center> 
   </div>
   </div>
   </div>
   <div class="fixed-bottom ">
      <div class="dark bg-dark " style="min-height: 40px;">
         <marquee style="color:#fff; margin-top: 7px;">
            <h9>Designed & Developed by Yash Sankhla, Student of Poornima Institute of Engineering and Technology</h9>
         </marquee>
      </div>
   </div>
   </body>
   """
st.markdown(html_temp,unsafe_allow_html=True)
