import streamlit as st 
from Home import app as face_rec
from streamlit_webrtc import webrtc_streamer
import av
import time
import cv2

st.subheader('Real-Time Attendance System')

with st.spinner('Retrieving Data from Redis DB ...'):    
    redis_face_db = face_rec.retrive_data(name='vattend:register')

waitTime = 30  
setTime = time.time()
realtimepred = face_rec.RealTimePred()  

def video_frame_callback(frame):
    global setTime
    
    img = frame.to_ndarray(format="bgr24") 
    pred_img = realtimepred.face_prediction(img, redis_face_db,
                                            'facial_features', 'regNo', thresh=0.5)
    
    timenow = time.time()
    difftime = timenow - setTime
    if difftime >= waitTime:
        realtimepred.saveLogs_redis()
        setTime = time.time()         
        print('Saved data to Redis database')

    for res in pred_img['faces']:
        x1, y1, x2, y2 = res['bbox'].astype(int)
        regNo = res['regNo']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, regNo, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(key="realtimePrediction", video_frame_callback=video_frame_callback)
