import streamlit as st 
from Home import app as face_rec
import numpy as np
import cv2

st.subheader('Image to reports')

with st.spinner('Retrieving Data ...'):    
    redis_face_db = face_rec.retrive_data(name='vattend:register')

st.success("Data successfully retrieved")

real_time_pred = face_rec.PredictionFromImage()

def process_images(uploaded_files):
    results = []
    for uploaded_file in uploaded_files:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        results.append(real_time_pred.face_prediction(img, redis_face_db, 'facial_features'))
    return results

uploaded_files = st.file_uploader("Choose images...", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

if uploaded_files:
    results = process_images(uploaded_files)
    for img in results:
        st.image(img, channels="BGR")
    real_time_pred.mark_attendance()
    st.success("Attendance marked successfully")
    face_rec.display_attendance_table()