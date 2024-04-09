import streamlit as st 
from Home import app as face_rec
from streamlit_webrtc import webrtc_streamer
import av

st.subheader('Predictions')

with st.spinner('Retrieving Data from Redis DB ...'):    
    redis_face_db = face_rec.retrive_data(name='vattend:register')

st.success("Data sucessfully retrived from Redis")

# Create an instance of RealTimePred class
real_time_pred = face_rec.RealTimePred(camera='outgoing')

def video_frame_callback(frame):
    global setTime
    
    img = frame.to_ndarray(format="bgr24") 
    pred_img = real_time_pred.face_prediction(img, redis_face_db, 'facial_features')
    
    # Call saveLogs_redis to update the logs
    real_time_pred.saveLogs_redis()
    
    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")

webrtc_streamer(key="realtimePrediction", video_frame_callback=video_frame_callback)
